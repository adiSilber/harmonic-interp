"""
Sampling with activation capture hooks.

This module provides a modified sample_batch function that captures
activations, KV-cache values, and logits during generation.
"""

import torch
import torch._inductor.config

from tqdm import tqdm
from pathlib import Path

from aria.inference import sample_min_p, sample_top_p
from aria.inference.model_cuda import TransformerLM
# ActivationCollector is vendored from aria (aria/inference/activation_hooks.py)
# into reproduce/activation_hooks.py; see that file's header.
from activation_hooks import ActivationCollector, CapturedActivations
from ariautils.tokenizer import Tokenizer, AbsTokenizer

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True

DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def update_seq_ids_(
    seq: torch.Tensor,
    idx: int,
    next_token_ids: torch.Tensor,
    dim_tok_inserted: list,
    eos_tok_seen: list,
    max_len: int,
    force_end: bool,
    tokenizer: Tokenizer,
):
    """Update sequence with next tokens (same as original)."""
    for _idx in range(seq.shape[0]):
        if eos_tok_seen[_idx] == True:
            next_token_ids[_idx] = tokenizer.tok_to_id[tokenizer.pad_tok]
        elif (
            force_end
            and idx >= max_len - 130
            and dim_tok_inserted[_idx] is False
            and tokenizer.id_to_tok[next_token_ids[_idx].item()][0]
            not in ("dur", "onset")
        ):
            next_token_ids[_idx] = tokenizer.tok_to_id[tokenizer.dim_tok]

        if next_token_ids[_idx] == tokenizer.tok_to_id[tokenizer.dim_tok]:
            dim_tok_inserted[_idx] = True
        elif next_token_ids[_idx] == tokenizer.tok_to_id[tokenizer.eos_tok]:
            eos_tok_seen[_idx] = True

    seq[:, idx] = next_token_ids


@torch.inference_mode()
def decode_one(
    model: TransformerLM,
    idxs: torch.Tensor,
    input_pos: torch.Tensor,
    pad_idxs: torch.Tensor | None = None,
) -> torch.Tensor:
    assert input_pos.shape[-1] == 1

    logits = model.forward(
        idxs=idxs,
        input_pos=input_pos,
        pad_idxs=pad_idxs,
    )[:, -1]

    return logits


@torch.inference_mode()
def prefill(
    model: TransformerLM,
    idxs: torch.Tensor,
    input_pos: torch.Tensor,
    pad_idxs: torch.Tensor | None = None,
) -> torch.Tensor:
    logits = model.forward(
        idxs=idxs,
        input_pos=input_pos,
        pad_idxs=pad_idxs,
    )

    return logits


@torch.autocast("cuda", dtype=DTYPE)
@torch.inference_mode()
def sample_batch_with_hooks(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: list,
    num_variations: int,
    max_new_tokens: int,
    temp: float,
    capture_positions: list[int] | None = None,
    capture_from_prompt_end: int = 5,
    capture_after_prompt: int = 10,
    force_end: bool = False,
    top_p: float | None = None,
    min_p: float | None = None,
    compile: bool = False,
    save_activations_to: str | None = None,
    capture_device: str = "cpu",
    hidden_states_positions: list[int] | None = None,
):
    """
    Sample from the model while capturing activations.

    Args:
        model: TransformerLM model
        tokenizer: Tokenizer instance
        prompt: List of prompt tokens
        num_variations: Number of variations to generate
        max_new_tokens: Maximum new tokens to generate
        temp: Sampling temperature
        capture_positions: Specific token positions to capture.
                          If None, auto-compute based on prompt end.
        capture_from_prompt_end: Capture this many tokens before prompt end
        capture_after_prompt: Capture this many tokens after prompt
        force_end: Force end token insertion
        top_p: Top-p sampling parameter
        min_p: Min-p sampling parameter
        compile: Whether to compile the model
        save_activations_to: Path to save captured activations
        capture_device: Device to store captured tensors ('cpu' recommended)

    Returns:
        tuple: (decoded_results, collectors_per_variation)
    """
    assert top_p is not None or min_p is not None
    assert 0.0 <= temp <= 2.0
    if top_p is not None:
        assert 0.5 <= top_p <= 1.0
    if min_p is not None:
        assert 0.0 <= min_p <= 1.0
    if force_end:
        assert max_new_tokens > 130, "prompt too long to use force_end=True"

    prompt_len = len(prompt)

    # Determine capture positions
    if capture_positions is None:
        # Capture around the prompt boundary (where the "change" happens)
        start_capture = max(0, prompt_len - capture_from_prompt_end)
        end_capture = prompt_len + capture_after_prompt
        capture_positions = list(range(start_capture, end_capture))

    model = model.cuda()
    model.eval()

    # Create activation collectors for each variation
    collectors = []
    for i in range(num_variations):
        collector = ActivationCollector(
            model,
            capture_hidden_states=True,
            capture_kv_cache=False,
            capture_logits=True,
            capture_positions=capture_positions,
            device=capture_device,
        )
        collector.register_hooks()
        collectors.append(collector)

    dim_tok_inserted = [False for _ in range(num_variations)]
    eos_tok_seen = [False for _ in range(num_variations)]
    total_len = prompt_len + max_new_tokens
    seq = torch.stack(
        [
            torch.tensor(
                tokenizer.encode(
                    prompt + [tokenizer.pad_tok] * (total_len - prompt_len)
                )
            )
            for _ in range(num_variations)
        ]
    ).cuda()

    if compile is True:
        global decode_one
        decode_one = torch.compile(
            decode_one,
            mode="reduce-overhead",
            fullgraph=True,
        )

    model.setup_cache(
        batch_size=num_variations,
        max_seq_len=total_len,
        dtype=DTYPE,
    )


    for idx in (
        pbar := tqdm(
            range(prompt_len, total_len),
            total=total_len - prompt_len,
            leave=False,
            disable=True,
        )
    ):
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            if idx == prompt_len:
                logits = prefill(
                    model,
                    idxs=seq[:, :idx],
                    input_pos=torch.arange(0, idx, device=seq.device),
                )[:, -1]

                # Capture activations for prompt positions AND first generated position
                # (position prompt_len's logits are computed during prefill)
                for batch_idx, collector in enumerate(collectors):
                    prompt_positions_to_capture = [p for p in capture_positions if p < prompt_len]
                    for i, pos in enumerate(prompt_positions_to_capture):
                        token_id = seq[batch_idx, pos].item()
                        collector.capture_step(
                            token_idx=pos,
                            token_id=token_id,
                            probs=None,  # No probs for prompt tokens
                            batch_idx=batch_idx,
                            seq_position=pos,  # Use actual position in sequence
                            clear_after=False,  # Don't clear yet
                        )
            else:
                logits = decode_one(
                    model,
                    idxs=seq[:, idx - 1 : idx],
                    input_pos=torch.tensor(
                        [(idx) - 1],
                        device=seq.device,
                        dtype=torch.int,
                    ),
                )

        if temp > 0.0:
            probs = torch.softmax(logits / temp, dim=-1)
            if min_p is not None:
                next_token_ids = sample_min_p(probs, min_p).flatten()
            else:
                next_token_ids = sample_top_p(probs, top_p).flatten()
        else:
            probs = torch.softmax(logits, dim=-1)
            next_token_ids = torch.argmax(logits, dim=-1).flatten()

        # Capture activations for generated positions
        # During decode_one at idx, we process the token at position idx-1
        # So the hidden states represent position idx-1, not idx
        # We capture position idx-1 with its token_id from the sequence
        for batch_idx, collector in enumerate(collectors):
            captured_pos = idx - 1  # The position we actually processed
            if idx > prompt_len and collector.should_capture(captured_pos):
                collector.capture_step(
                    token_idx=captured_pos,
                    token_id=seq[batch_idx, captured_pos].item(),
                    probs=probs,
                    batch_idx=batch_idx,
                )

        update_seq_ids_(
            seq=seq,
            idx=idx,
            next_token_ids=next_token_ids,
            dim_tok_inserted=dim_tok_inserted,
            eos_tok_seen=eos_tok_seen,
            max_len=total_len,
            force_end=force_end,
            tokenizer=tokenizer,
        )

        if all(seen_eos is True for seen_eos in eos_tok_seen):
            break

    # Remove hooks
    for collector in collectors:
        collector.clear_hooks()

    # Save activations if path provided
    if save_activations_to is not None:
        save_path = Path(save_activations_to)
        if len(collectors) == 1:
            collectors[0].save(save_path, tokenizer=tokenizer,
                               hidden_states_positions=hidden_states_positions)
        else:
            for i, collector in enumerate(collectors):
                collector.save(save_path / f'variation_{i}', tokenizer=tokenizer,
                               hidden_states_positions=hidden_states_positions)

    decoded_results = [tokenizer.decode(s) for s in seq.tolist()]
    decoded_results = [
        (
            res[: res.index(tokenizer.eos_tok) + 1]
            if tokenizer.eos_tok in res
            else res
        )
        for res in decoded_results
    ]

    return decoded_results, collectors



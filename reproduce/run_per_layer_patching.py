#!/usr/bin/env python3
"""
Per-layer residual-stream patching.

For each (chorale, cut_tick, seed) that has minorized_activations, and for
each transformer layer i, patch the full residual stream at the last prompt
position (prompt_len-1) after layer i with the SAVED minorized hidden state
at that layer+position, then generate a full continuation.

Nothing is removed — only new directories are written:
    seed_N/per_layer_last_position_patching/layer_<i>_patch/
        output.mid
        activations/   (hidden states all-layers + logits + probs at
                        patch_pos and first 20 generated positions)

KV cache for positions 0…patch_pos-1 is recomputed once per seed with a single
prefill forward pass over the prompt tokens, then reused (restored from an
in-memory snapshot) for every layer patch. This is exact — the prompt is
unchanged across layers, so the same context KV is deterministically produced —
and avoids both saving large kv_cache.pt files in step 3 and re-running it.

Prompt token IDs are recovered by tokenizing the source cut MIDI
(found via data_dir in experiment_info.json) — NOT original.mid, which
contains the generated continuation and produces wrong token ordering near
the cut boundary.

Usage:
    python reproduce/run_per_layer_patching.py \
        --experiment_dir reproduce_output/patching/<timestamp>_reproduce_major
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from consts import (
    TEMP, MIN_P,
    CHECKPOINT, MODEL_CONFIG_NAME,
    DTYPE,
)

CAPTURE_GENERATED_TOKENS = 20
PATCH_GEN_LENGTH = 100   # tokens to generate after patching
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── model ──────────────────────────────────────────────────────────────────────

def _set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model_and_tokenizer():
    from aria.config import load_model_config
    from aria.inference.model_cuda import TransformerLM
    from aria.model import ModelConfig
    from ariautils.tokenizer import AbsTokenizer
    from safetensors.torch import load_file
    cfg = ModelConfig(**load_model_config(name=MODEL_CONFIG_NAME))
    cfg.set_vocab_size(AbsTokenizer().vocab_size)
    model = TransformerLM(cfg)
    model.load_state_dict(load_file(CHECKPOINT), strict=False)
    model = model.to(DEVICE).eval()
    tokenizer = AbsTokenizer()
    print(f"Model: {len(model.model.encode_layers)} layers")
    return model, tokenizer


# ── data helpers ───────────────────────────────────────────────────────────────

def _get_patch_pos(minor_meta_path, n_gen):
    """Derive last prompt position = first-generated-position - 1."""
    with open(minor_meta_path) as f:
        positions = json.load(f)["capture_positions"]
    if len(positions) <= n_gen:
        raise ValueError(f"{len(positions)} captured positions <= n_gen={n_gen}")
    return positions[-n_gen] - 1   # = prompt_len - 1


def _get_patch_token_id(seed_dir, patch_pos):
    """Read the token ID at patch_pos from original_activations/metadata.json."""
    meta_path = os.path.join(seed_dir, "original_activations", "metadata.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    try:
        idx = meta["capture_positions"].index(patch_pos)
        return meta["token_ids"][idx]
    except ValueError:
        return None


def _load_minor_data(seed_dir, n_gen):
    """Returns (patch_pos, minor_hs) or None."""
    meta = os.path.join(seed_dir, "minorized_activations", "metadata.json")
    if not os.path.exists(meta):
        return None
    patch_pos = _get_patch_pos(meta, n_gen)
    hs_path = os.path.join(seed_dir, "minorized_activations",
                           f"position_{patch_pos}", "hidden_states.pt")
    if not os.path.exists(hs_path):
        return None
    minor_hs = torch.load(hs_path, map_location="cpu", weights_only=True)
    return patch_pos, minor_hs


def _find_prompt_midi(seed_dir, exp_info):
    """Locate the source cut MIDI from experiment_info data_dir.

    seed_dir: .../experiment_dir/chorale_XXXX/cut_tick_YYY/seed_N
    prompt MIDI: data_dir/chorale_XXXX_cuts_output/chorale_XXXX_cut_tick_YYY.mid
    """
    data_dir = exp_info.get("data_dir")
    if not data_dir:
        return None
    p = Path(seed_dir)
    cut_tick = p.parent.name        # e.g. "cut_tick_220"
    chorale = p.parent.parent.name  # e.g. "chorale_0001"
    prompt_midi = os.path.join(
        data_dir, f"{chorale}_cuts_output", f"{chorale}_{cut_tick}.mid"
    )
    return prompt_midi if os.path.exists(prompt_midi) else None


def _load_prompt_ids(seed_dir, prompt_len, tokenizer, exp_info):
    """Tokenize the source cut MIDI to recover prompt token IDs.

    Uses the original cut MIDI (not original.mid) to avoid re-tokenization
    artifacts: original.mid contains the generated continuation, so re-
    tokenizing it interleaves generated notes with prompt notes near the cut
    boundary and produces a different token sequence at patch_pos.
    """
    from ariautils.midi import MidiDict
    prompt_midi = _find_prompt_midi(seed_dir, exp_info)
    if prompt_midi is None:
        return None
    midi_dict = MidiDict.from_midi(prompt_midi)
    tokens = tokenizer.tokenize(midi_dict, add_dim_tok=False, add_eos_tok=False)
    ids = tokenizer.encode(tokens)
    if len(ids) < prompt_len:
        print(f"      prompt MIDI too short ({len(ids)}) for prompt_len={prompt_len}")
        return None
    return ids[:prompt_len]


@torch.autocast(DEVICE, dtype=DTYPE)
@torch.inference_mode()
def _compute_kv_snapshot(model, prompt_ids, patch_pos):
    """Recompute the prompt KV for positions 0..patch_pos-1 with one prefill.

    Requires the KV cache to already be set up (see _setup_cache). Runs a single
    forward pass over the prompt context, then snapshots each layer's k/v cache
    so it can be restored before every layer patch. The prompt is identical
    across all layer patches, so this snapshot is computed once per seed.
    Returns a list of per-layer {"k", "v"}.
    """
    from aria.inference.sample_cuda import prefill
    ctx = torch.tensor([prompt_ids[:patch_pos]], dtype=torch.long, device=DEVICE)
    input_pos = torch.arange(patch_pos, dtype=torch.int, device=DEVICE)
    prefill(model, idxs=ctx, input_pos=input_pos)
    return [{"k": layer.kv_cache.k_cache[0, :, :patch_pos].clone(),
             "v": layer.kv_cache.v_cache[0, :, :patch_pos].clone()}
            for layer in model.model.encode_layers]


# ── cache setup ────────────────────────────────────────────────────────────────

def _setup_cache(model, batch_size, max_seq_len, dtype):
    from aria.inference.model_cuda import KVCache, precompute_freqs_cis
    device = next(model.parameters()).device
    for b in model.model.encode_layers:
        b.kv_cache = KVCache(
            max_batch_size=batch_size,
            max_seq_length=max_seq_len,
            n_heads=model.model_config.n_heads,
            head_dim=model.model_config.d_model // model.model_config.n_heads,
            dtype=dtype,
        ).to(device)
    model.model.freqs_cis = precompute_freqs_cis(
        seq_len=max_seq_len,
        n_elem=model.model_config.d_model // model.model_config.n_heads,
        base=500000,
        dtype=dtype,
    ).to(device)
    model.model.causal_mask = torch.tril(
        torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)
    ).to(device)


# ── KV snapshot ────────────────────────────────────────────────────────────────

def _restore_kv_snapshot(model, snap):
    n = snap[0]["k"].shape[1]
    for layer, kv in zip(model.model.encode_layers, snap):
        layer.kv_cache.k_cache[0, :, :n].copy_(kv["k"])
        layer.kv_cache.v_cache[0, :, :n].copy_(kv["v"])


# ── patching hook ──────────────────────────────────────────────────────────────

def _make_patching_hook(patch_h):
    def hook(module, inp, output):
        patched = output.clone()
        patched[0, 0, :] = patch_h
        return patched
    return hook


# ── decode helpers ─────────────────────────────────────────────────────────────

def _sdpa_decode(model, seq, pos):
    ip = torch.tensor([pos], device=seq.device, dtype=torch.int)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        from aria.inference.sample_cuda import decode_one
        return decode_one(model, idxs=seq[:, pos:pos + 1], input_pos=ip)


def _patched_decode_step(model, seq, patch_pos, layer_i, minor_h, collector):
    # Register patching hook FIRST so it fires before the collector's hook.
    # PyTorch passes the patched output to subsequent hooks, so the collector
    # stores the patched value (with .clone() to survive memory reuse).
    handle = model.model.encode_layers[layer_i].register_forward_hook(
        _make_patching_hook(minor_h)
    )
    collector.register_hooks()
    logits = _sdpa_decode(model, seq, patch_pos)
    handle.remove()
    return logits


# ── generation loop ────────────────────────────────────────────────────────────

def _maybe_capture(collector, seq, idx, probs, capture_set):
    if idx > 0 and (idx - 1) in capture_set:
        collector.capture_step(token_idx=idx - 1, token_id=seq[0, idx - 1].item(),
                               probs=probs, batch_idx=0)


def _generation_loop(model, tokenizer, seq, prompt_len, total_len,
                     first_logits, collector, capture_set):
    from aria.inference import sample_min_p
    from aria.inference.sample_cuda import update_seq_ids_
    dim_done, eos_done = [False], [False]
    for idx in range(prompt_len, total_len):
        logits = first_logits if idx == prompt_len else _sdpa_decode(model, seq, idx - 1)
        probs = torch.softmax(logits / TEMP, dim=-1)
        next_ids = sample_min_p(probs, MIN_P).flatten()
        _maybe_capture(collector, seq, idx, probs, capture_set)
        update_seq_ids_(seq=seq, idx=idx, next_token_ids=next_ids,
                        dim_tok_inserted=dim_done, eos_tok_seen=eos_done,
                        max_len=total_len, force_end=False, tokenizer=tokenizer)
        if all(eos_done):
            break


# ── save ───────────────────────────────────────────────────────────────────────

def _save_layer_outputs(tokenizer, seq, collector, out_dir, patch_pos):
    os.makedirs(out_dir, exist_ok=True)
    decoded = tokenizer.decode(seq[0].tolist())
    if tokenizer.eos_tok in decoded:
        decoded = decoded[:decoded.index(tokenizer.eos_tok) + 1]
    tokenizer.detokenize(decoded).to_midi().save(os.path.join(out_dir, "output.mid"))

    hs_positions = {patch_pos, patch_pos + 1}
    act_path = Path(out_dir) / "activations"
    act_path.mkdir(parents=True, exist_ok=True)
    from utils import build_token_translation
    token_ids = [c.token_id for c in collector.captured_data]
    metadata = {
        "num_captures": len(collector.captured_data),
        "capture_positions": [c.token_idx for c in collector.captured_data],
        "token_ids": token_ids,
        "token_translation": build_token_translation(token_ids, tokenizer),
    }
    with open(act_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    for captured in collector.captured_data:
        cap_dir = act_path / f"position_{captured.token_idx}"
        cap_dir.mkdir(exist_ok=True)
        if captured.hidden_states and captured.token_idx in hs_positions:
            torch.save({f"layer_{k}": v for k, v in captured.hidden_states.items()},
                       cap_dir / "hidden_states.pt")
        if captured.logits is not None:
            torch.save(captured.logits, cap_dir / "logits.pt")
        if captured.probs is not None:
            torch.save(captured.probs, cap_dir / "probs.pt")


# ── one layer experiment ───────────────────────────────────────────────────────

@torch.autocast(DEVICE, dtype=DTYPE)
@torch.inference_mode()
def _run_layer(model, tokenizer, prompt_ids, patch_token_id, gen_length, patch_pos,
               layer_i, minor_h, kv_snap, out_dir, seed):
    # ActivationCollector vendored from aria (see reproduce/activation_hooks.py header)
    from activation_hooks import ActivationCollector
    _set_seed(seed)
    _restore_kv_snapshot(model, kv_snap)

    pad_id = tokenizer.encode([tokenizer.pad_tok])[0]
    prompt_len = patch_pos + 1
    seq = torch.full((1, prompt_len + gen_length), pad_id,
                     dtype=torch.long).to(DEVICE)
    seq[0, :prompt_len] = torch.tensor(prompt_ids, dtype=torch.long).to(DEVICE)
    seq[0, patch_pos] = patch_token_id

    gen_cap_pos = list(range(prompt_len, prompt_len + CAPTURE_GENERATED_TOKENS))
    cap_pos = [patch_pos] + gen_cap_pos
    collector = ActivationCollector(model, capture_hidden_states=True, capture_kv_cache=False,
                                    capture_logits=True, capture_positions=cap_pos, device="cpu")
    first_logits = _patched_decode_step(model, seq, patch_pos, layer_i,
                                        minor_h.to(device=DEVICE), collector)
    first_probs = torch.softmax(first_logits.float() / TEMP, dim=-1)
    collector.capture_step(token_idx=patch_pos, token_id=seq[0, patch_pos].item(),
                           probs=first_probs, batch_idx=0, seq_position=-1)
    _generation_loop(model, tokenizer, seq, prompt_len, prompt_len + gen_length,
                     first_logits, collector, set(gen_cap_pos))
    collector.clear_hooks()
    _save_layer_outputs(tokenizer, seq, collector, out_dir, patch_pos)


# ── traversal ─────────────────────────────────────────────────────────────────

def _is_done(layer_out):
    return (os.path.exists(os.path.join(layer_out, "output.mid"))
            and os.path.exists(os.path.join(layer_out, "activations", "metadata.json")))


def _run_all_layers(model, tokenizer, prompt_ids, patch_token_id, gen_length, patch_pos,
                    minor_hs, kv_snap, patch_base, seed):
    for layer_i in range(len(model.model.encode_layers)):
        layer_out = os.path.join(patch_base, f"layer_{layer_i}_patch")
        if _is_done(layer_out):
            print(f"      layer {layer_i:2d}: already done")
            continue
        minor_h = minor_hs.get(f"layer_{layer_i}")
        if minor_h is None:
            continue
        print(f"      layer {layer_i:2d}: running…", end="", flush=True)
        _run_layer(model, tokenizer, prompt_ids, patch_token_id, gen_length, patch_pos,
                   layer_i, minor_h, kv_snap, layer_out, seed)
        print(" done")


def process_seed_dir(model, tokenizer, seed_dir, exp_info):
    seed = int(os.path.basename(seed_dir).replace("seed_", ""))
    n_gen = exp_info.get("capture_generated_tokens", CAPTURE_GENERATED_TOKENS)

    minor = _load_minor_data(seed_dir, n_gen)
    if minor is None:
        print("      skipping (missing minor data)"); return
    patch_pos, minor_hs = minor

    patch_token_id = _get_patch_token_id(seed_dir, patch_pos)
    if patch_token_id is None:
        print("      skipping (patch_pos not found in original_activations/metadata.json)")
        return

    prompt_len = patch_pos + 1

    prompt_ids = _load_prompt_ids(seed_dir, prompt_len, tokenizer, exp_info)
    if prompt_ids is None:
        print("      skipping (could not find/tokenize source cut MIDI)")
        return

    print(f"      patch_pos={patch_pos}  prompt_len={prompt_len}  gen_len={PATCH_GEN_LENGTH}")
    patch_base = os.path.join(seed_dir, "per_layer_last_position_patching")
    # Set up the KV cache, then recompute the prompt context once via prefill.
    _setup_cache(model, batch_size=1, max_seq_len=prompt_len + PATCH_GEN_LENGTH, dtype=DTYPE)
    kv_snap = _compute_kv_snapshot(model, prompt_ids, patch_pos)
    _run_all_layers(model, tokenizer, prompt_ids, patch_token_id, PATCH_GEN_LENGTH, patch_pos,
                    minor_hs, kv_snap, patch_base, seed)


def process_experiment(experiment_dir):
    with open(os.path.join(experiment_dir, "experiment_info.json")) as f:
        exp_info = json.load(f)
    model, tokenizer = load_model_and_tokenizer()
    chorales = sorted(d for d in os.listdir(experiment_dir)
                      if d.startswith("chorale_") and os.path.isdir(os.path.join(experiment_dir, d)))
    for chorale in chorales:
        chorale_dir = os.path.join(experiment_dir, chorale)
        cut_ticks = sorted(d for d in os.listdir(chorale_dir)
                           if d.startswith("cut_tick_") and os.path.isdir(os.path.join(chorale_dir, d)))
        for cut_tick in cut_ticks:
            print(f"\n{chorale}/{cut_tick}")
            seed_entries = sorted(d for d in os.listdir(os.path.join(chorale_dir, cut_tick))
                                  if d.startswith("seed_") and os.path.isdir(os.path.join(chorale_dir, cut_tick, d)))
            for seed_entry in seed_entries:
                print(f"    {seed_entry}")
                process_seed_dir(model, tokenizer,
                                 os.path.join(chorale_dir, cut_tick, seed_entry),
                                 exp_info)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_dir", required=True)
    args = p.parse_args()
    process_experiment(args.experiment_dir)


if __name__ == "__main__":
    main()

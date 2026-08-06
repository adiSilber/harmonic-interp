#!/usr/bin/env python3
"""
Activation-capture generation for a single chorale (per-layer patching, phase 2).

Usage:
    python reproduce/run_single_chorale_hooks.py --chorale chorale_0000 \
        --data_dir data/major_chorale_corpus

For each cut it runs activation-capturing generation on the original (major)
prompt and its minorized counterpart, saving:
  - the residual stream (all layers) at the single patch position (prompt_len-1)
  - logits/probs at the first N generated positions
No KV cache is saved. `extract_patching_activations.py` is the batch driver over many chorales.
"""

import argparse
import json
import os
import sys
import datetime
import random

# Ensure this reproduce/ dir is on sys.path so the sibling modules import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from consts import (
    SEEDS, CONTINUATION_SECONDS, TEMP, MIN_P,
    CHECKPOINT, MODEL_CONFIG_NAME, MIN_GEN_LENGTH, MAX_GEN_LENGTH, DTYPE,
    CAPTURE_GENERATED_TOKENS,
)
from utils import midi_to_note_name

SEED = SEEDS[0]  # default seed for single-chorale runs

# ── Helper functions ─────────────────────────────────────────────────────────

def format_token(tok, idx: int, capture_positions: set) -> str:
    """Format a token for printing with note name if it's a piano token."""
    marker = " *" if idx in capture_positions else ""
    if isinstance(tok, tuple) and tok[0] == 'piano':
        note_name = midi_to_note_name(tok[1])
        return f"{idx:4d}: {tok}  [{note_name}]{marker}"
    else:
        return f"{idx:4d}: {tok}{marker}"


def _set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model_and_tokenizer():
    """Load the Aria model and tokenizer."""
    from aria.config import load_model_config
    from aria.inference.model_cuda import TransformerLM
    from aria.model import ModelConfig
    from ariautils.tokenizer import AbsTokenizer
    from safetensors.torch import load_file

    print("Loading model...")
    model_config = ModelConfig(**load_model_config(name=MODEL_CONFIG_NAME))
    model_config.set_vocab_size(AbsTokenizer().vocab_size)
    model = TransformerLM(model_config)

    state_dict = load_file(CHECKPOINT)
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda()
    model.eval()

    tokenizer = AbsTokenizer()

    print("Model loaded successfully")
    return model, tokenizer


def compute_generation_length(midi_path: str, tokenizer) -> int:
    """Compute token generation length targeting CONTINUATION_SECONDS of audio."""
    from ariautils.midi import MidiDict

    midi_dict = MidiDict.from_midi(midi_path)
    tokens = tokenizer.tokenize(midi_dict, add_dim_tok=False, add_eos_tok=False)

    if not midi_dict.note_msgs:
        return MIN_GEN_LENGTH

    max_end_ms = max(
        midi_dict.tick_to_ms(msg["data"]["end"]) for msg in midi_dict.note_msgs
    )
    duration_seconds = max_end_ms / 1000.0

    if duration_seconds <= 0:
        return MIN_GEN_LENGTH

    tokens_per_second = len(tokens) / duration_seconds
    gen_length = int(tokens_per_second * CONTINUATION_SECONDS)
    gen_length = max(MIN_GEN_LENGTH, min(MAX_GEN_LENGTH, gen_length))

    return gen_length


# ── Single-file generation with activation capture ───────────────────────────

def run_single_file(
    model,
    tokenizer,
    midi_path: str,
    output_dir: str,
    label: str,
    seed: int,
    capture_generated_tokens: int = CAPTURE_GENERATED_TOKENS,
):
    """Run generation with hooks on a single MIDI file.

    For the per-layer patching experiment we capture:
      - the single patch position (last prompt token, prompt_len-1): the residual
        stream is saved here (all layers) — it is the only place we patch.
      - the first `capture_generated_tokens` generated positions: logits/probs are
        saved here to measure the generated continuation.
    """
    from ariautils.midi import MidiDict
    from sample_with_hooks import sample_batch_with_hooks

    _set_seed(seed)

    gen_length = compute_generation_length(midi_path, tokenizer)

    # Load and tokenize
    midi_dict = MidiDict.from_midi(midi_path)
    prompt = tokenizer.tokenize(midi_dict, add_dim_tok=False, add_eos_tok=False)

    # Capture set: the patch position (last prompt token) + the first N generated
    prompt_len = len(prompt)
    patch_pos = prompt_len - 1
    capture_positions = [patch_pos] + list(
        range(prompt_len, prompt_len + capture_generated_tokens)
    )
    # Hidden states (residual stream) are only needed at the patch position.
    hidden_states_positions = [patch_pos]

    # Paths
    midi_output = os.path.join(output_dir, f"{label}.mid")
    activations_output = os.path.join(output_dir, f"{label}_activations")

    os.makedirs(output_dir, exist_ok=True)

    # Run with hooks
    results, collectors = sample_batch_with_hooks(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        num_variations=1,
        max_new_tokens=gen_length,
        temp=TEMP,
        min_p=MIN_P,
        top_p=None,
        capture_positions=capture_positions,
        save_activations_to=activations_output,
        capture_device="cpu",
        hidden_states_positions=hidden_states_positions,
    )

    # Save MIDI
    result_tokens = results[0]
    result_midi = tokenizer.detokenize(result_tokens)
    result_midi.to_midi().save(midi_output)

    return collectors[0]


# ── Single-chorale entry point ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run activation capture on a single chorale (original + minorized)"
    )
    parser.add_argument("--chorale", default="chorale_0000",
                        help="Chorale name (default: chorale_0000)")
    parser.add_argument("--data_dir", default="data/major_chorale_corpus",
                        help="Data directory (default: data/major_chorale_corpus)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Random seed (default: {SEED})")
    parser.add_argument("--capture_generated_tokens", type=int,
                        default=CAPTURE_GENERATED_TOKENS,
                        help=f"Capture first N generated positions "
                             f"(default: {CAPTURE_GENERATED_TOKENS})")
    parser.add_argument("--cut_tick", type=int, default=None,
                        help="Only process this cut tick (e.g. 56). Default: all cuts.")
    args = parser.parse_args()

    chorale_dir = os.path.join(args.data_dir, f"{args.chorale}_cuts_output")
    if not os.path.isdir(chorale_dir):
        print(f"Error: Chorale directory not found: {chorale_dir}")
        sys.exit(1)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("reproduce_output", "patching",
                              f"{timestamp}_{args.chorale}_hooks")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "experiment_info.json"), "w") as f:
        json.dump({
            "chorale": args.chorale,
            "timestamp": timestamp,
            "data_dir": os.path.abspath(args.data_dir),
            "seed": args.seed,
            "capture_generated_tokens": args.capture_generated_tokens,
            "temp": TEMP,
            "min_p": MIN_P,
        }, f, indent=2)

    print(f"Output directory: {output_dir}")

    model, tokenizer = load_model_and_tokenizer()

    midi_files = sorted(
        f for f in os.listdir(chorale_dir)
        if f.endswith(".mid") and os.path.isfile(os.path.join(chorale_dir, f))
    )
    minorized_dir = os.path.join(chorale_dir, "minorized")

    n_cuts = 0
    for midi_file in midi_files:
        base = midi_file.replace(".mid", "")
        idx = base.find("cut_tick_")
        if idx == -1:
            continue
        tick_label = base[idx:]
        tick_num = int(tick_label.replace("cut_tick_", ""))
        if args.cut_tick is not None and tick_num != args.cut_tick:
            continue

        original_path = os.path.join(chorale_dir, midi_file)
        seed_out_dir = os.path.join(output_dir, tick_label, f"seed_{args.seed}")

        run_single_file(
            model=model, tokenizer=tokenizer, midi_path=original_path,
            output_dir=seed_out_dir, label="original", seed=args.seed,
            capture_generated_tokens=args.capture_generated_tokens,
        )

        minor_path = os.path.join(minorized_dir, base + "_minor.mid")
        if os.path.isfile(minor_path):
            run_single_file(
                model=model, tokenizer=tokenizer, midi_path=minor_path,
                output_dir=seed_out_dir, label="minorized", seed=args.seed,
                capture_generated_tokens=args.capture_generated_tokens,
            )
        n_cuts += 1

    print(f"\n{'='*60}")
    print(f"Complete. Processed {n_cuts} cut points.")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()

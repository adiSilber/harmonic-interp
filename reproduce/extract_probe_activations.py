#!/usr/bin/env python3
"""
Extract residual-stream activations for the probing experiments.

For every bar-truncated MIDI (chorale_XXXX_bars_1-N.mid) in
probes_data/<split>/chorale_XXXX/, runs one forward pass through Aria and saves
the residual stream at the LAST token of the sequence, for all layers.

Output, one file per chorale dir: activations.pt
    {'activations': float32 tensor [n_bars, n_layers, d_model],
     'bar_indices': [1, 2, ..., n_bars]}

This is the format read by the probe training script (bar-end position only;
the optional activations_pos-2/-3/-4.pt variants are not produced).

REQUIRES A GPU.

Usage:
    python reproduce/extract_probe_activations.py                    # all splits
    python reproduce/extract_probe_activations.py --splits train test
    python reproduce/extract_probe_activations.py --out_root /tmp/x  # write elsewhere
"""

import argparse
import os
import re
import sys

import torch

# Ensure this reproduce/ dir is on sys.path so the sibling modules import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from consts import CHECKPOINT, MODEL_CONFIG_NAME, DTYPE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LEN = 4096
BAR_MIDI_RE = re.compile(r"_bars_1-(\d+)\.mid$")


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
    print(f"Model: {len(model.model.encode_layers)} layers, device={DEVICE}")
    return model, tokenizer


def _setup_cache(model, max_seq_len):
    from aria.inference.model_cuda import KVCache, precompute_freqs_cis
    for b in model.model.encode_layers:
        b.kv_cache = KVCache(
            max_batch_size=1,
            max_seq_length=max_seq_len,
            n_heads=model.model_config.n_heads,
            head_dim=model.model_config.d_model // model.model_config.n_heads,
            dtype=DTYPE,
        ).to(DEVICE)
    model.model.freqs_cis = precompute_freqs_cis(
        seq_len=max_seq_len,
        n_elem=model.model_config.d_model // model.model_config.n_heads,
        base=500000,
        dtype=DTYPE,
    ).to(DEVICE)
    model.model.causal_mask = torch.tril(
        torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)
    ).to(DEVICE)


def tokenize_midi(tokenizer, midi_path):
    from ariautils.midi import MidiDict
    midi_dict = MidiDict.from_midi(midi_path)
    tokens = tokenizer.tokenize(midi_dict, add_dim_tok=False, add_eos_tok=False)
    return tokenizer.encode(tokens)


@torch.autocast(DEVICE, dtype=DTYPE)
@torch.inference_mode()
def last_token_activations(model, ids):
    """Residual stream at the last token of `ids`: float32 [n_layers, d_model]."""
    from aria.inference.sample_cuda import prefill
    feats = {}

    def make_hook(layer_idx):
        def hook(module, inp, out):
            feats[layer_idx] = out[0, -1, :].float().cpu()
        return hook

    handles = [layer.register_forward_hook(make_hook(i))
               for i, layer in enumerate(model.model.encode_layers)]
    try:
        seq = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        input_pos = torch.arange(len(ids), dtype=torch.int, device=DEVICE)
        prefill(model, idxs=seq, input_pos=input_pos)
    finally:
        for h in handles:
            h.remove()
    return torch.stack([feats[i] for i in range(len(feats))])


def bar_midis(chorale_dir):
    """Sorted [(bar_number, midi_path), ...] for one chorale dir."""
    out = []
    for f in os.listdir(chorale_dir):
        m = BAR_MIDI_RE.search(f)
        if m:
            out.append((int(m.group(1)), os.path.join(chorale_dir, f)))
    return sorted(out)


def process_chorale(model, tokenizer, chorale_dir, out_path):
    bars = bar_midis(chorale_dir)
    if not bars:
        print(f"    {os.path.basename(chorale_dir)}: no bar MIDIs, skipping")
        return
    rows, indices = [], []
    for bar_num, midi_path in bars:
        ids = tokenize_midi(tokenizer, midi_path)
        if not ids or len(ids) > MAX_SEQ_LEN:
            print(f"    {os.path.basename(midi_path)}: {len(ids)} tokens, skipping")
            continue
        rows.append(last_token_activations(model, ids))
        indices.append(bar_num)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"activations": torch.stack(rows), "bar_indices": indices}, out_path)
    print(f"    {os.path.basename(chorale_dir)}: {len(indices)} bars -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probes_data", default="reproduce_output/probes_data",
                   help="Root dir with <split>/chorale_XXXX/ bar-truncated MIDIs (Step 7 output)")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--out_root", default=None,
                   help="Write activations under this root instead of --probes_data")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N chorales per split (smoke test)")
    args = p.parse_args()
    out_root = args.out_root or args.probes_data

    model, tokenizer = load_model_and_tokenizer()
    _setup_cache(model, MAX_SEQ_LEN)

    for split in args.splits:
        split_dir = os.path.join(args.probes_data, split)
        if not os.path.isdir(split_dir):
            print(f"{split}: {split_dir} not found, skipping")
            continue
        chorales = sorted(d for d in os.listdir(split_dir)
                          if d.startswith("chorale_")
                          and os.path.isdir(os.path.join(split_dir, d)))
        if args.limit:
            chorales = chorales[:args.limit]
        print(f"\n=== {split}: {len(chorales)} chorales ===")
        for chorale in chorales:
            out_path = os.path.join(out_root, split, chorale, "activations.pt")
            if os.path.exists(out_path) and not args.overwrite:
                print(f"    {chorale}: exists, skipping")
                continue
            process_chorale(model, tokenizer,
                            os.path.join(split_dir, chorale), out_path)


if __name__ == "__main__":
    main()

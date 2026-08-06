#!/usr/bin/env python3
"""
Build the steering directions for the mode / relative-minor / parallel-minor
conditions (paper Section 5.3, Eq. 4), from the trained probes.

For a source major key and a target class, the raw direction at layer L is the
difference of the two probe-weight rows:

    mode      v_L = W_mode[L, minor] - W_mode[L, major]        (toward minor)
    relative  v_L = W_key[L, relative_minor(key)] - W_key[L, key]
    parallel  v_L = W_key[L, parallel_minor(key)] - W_key[L, key]

Eq. 4 then scales the *unit* direction to the typical hidden-state magnitude of
the layer:

    g_L = ||h_bar_L|| * v_L / ||v_L||

so that at generation time the applied vector is simply  alpha * g_L.

The magnitude ||h_bar_L|| (mean residual-stream norm at layer L) is computed from
the TRAINING split (the Step-8 probes_data/train activations).

Directions are built for every source major key defined in the mapping tables
(independent of which chorales we later steer). Output:
    reproduce_output/steering/directions.pt
        {'h_norm_mean_per_layer': [16],
         'mode':     {layer: tensor[d_model]},
         'relative': {major_key: {layer: tensor[d_model]}},
         'parallel': {major_key: {layer: tensor[d_model]}}}

CPU-only. Run after Step 8 (probe activations) and Step 9 (train probes).

Usage:
    python reproduce/build_steering_directions.py
"""

import argparse
import glob
import os

import torch

N_LAYERS = 16

# Source major key -> target key (from steering_sweep.py mapping tables).
RELATIVE_MINOR = {
    "A major": "F# minor", "A- major": "F minor", "B- major": "G minor",
    "C major": "A minor",  "D major": "B minor",  "E major": "C# minor",
    "E- major": "C minor", "F major": "D minor",  "G major": "E minor",
}
PARALLEL_MINOR = {
    "A major": "A minor", "C major": "C minor", "D major": "D minor",
    "E major": "E minor", "F major": "F minor", "G major": "G minor",
}


def mean_hidden_norm_per_layer(train_dir):
    """||h_bar_L|| for L=0..15, averaged over every train bar's last-token activation."""
    sums = torch.zeros(N_LAYERS, dtype=torch.float64)
    n = 0
    paths = sorted(glob.glob(os.path.join(train_dir, "chorale_*", "activations.pt")))
    if not paths:
        raise SystemExit(f"No train activations under {train_dir} (run Step 8 first).")
    for p in paths:
        a = torch.load(p, map_location="cpu", weights_only=False)["activations"].float()
        sums += a.norm(dim=-1).sum(dim=0).double()   # [n_bars,16] -> sum over bars
        n += a.shape[0]
    return (sums / n).float(), n, len(paths)


def scaled_direction(W, idx, src_key, tgt_key, h_norm):
    """g_L = ||h_bar_L|| * (W[L,tgt] - W[L,src]) / ||W[L,tgt] - W[L,src]||, per layer."""
    out = {}
    for L in range(N_LAYERS):
        v = W[L, idx[tgt_key]] - W[L, idx[src_key]]
        out[L] = (h_norm[L] * v / v.norm()).clone()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probes", default="reproduce_output/probes/probe_weights.pt")
    p.add_argument("--train_acts", default="reproduce_output/probes_data/train")
    p.add_argument("--output", default="reproduce_output/steering/directions.pt")
    args = p.parse_args()

    probes = torch.load(args.probes, map_location="cpu", weights_only=False)

    print("Mean hidden-state norm per layer (TRAIN split only)...")
    h_norm, n_bars, n_chorales = mean_hidden_norm_per_layer(args.train_acts)
    print(f"  {n_chorales} chorales, {n_bars} bars")
    print("  ||h_bar_L|| =", [round(x, 1) for x in h_norm.tolist()])

    # --- mode: single direction, minor - major ---
    mode_W = probes["mode"]["W"]
    mode_cls = probes["mode"]["classes"]
    m_idx = {c: i for i, c in enumerate(mode_cls)}
    mode_dir = {}
    for L in range(N_LAYERS):
        v = mode_W[L, m_idx["minor"]] - mode_W[L, m_idx["major"]]
        mode_dir[L] = (h_norm[L] * v / v.norm()).clone()

    # --- relative / parallel: one direction per source major key ---
    key_W = probes["key"]["W"]
    key_cls = probes["key"]["classes"]
    k_idx = {c: i for i, c in enumerate(key_cls)}

    def build(mapping):
        out = {}
        for src, tgt in mapping.items():
            if src not in k_idx or tgt not in k_idx:
                print(f"  skip {src} -> {tgt} (class not in probe)")
                continue
            out[src] = scaled_direction(key_W, k_idx, src, tgt, h_norm)
        return out

    relative = build(RELATIVE_MINOR)
    parallel = build(PARALLEL_MINOR)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({"h_norm_mean_per_layer": h_norm,
                "mode": mode_dir, "relative": relative, "parallel": parallel},
               args.output)

    print(f"\nBuilt directions:  mode=1  relative={len(relative)}  parallel={len(parallel)}")
    print(f"  relative keys: {sorted(relative)}")
    print(f"  parallel keys: {sorted(parallel)}")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()

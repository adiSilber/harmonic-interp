#!/usr/bin/env python3
"""
Train the linear probes (paper Figures 4 and 6).

Four probe tasks, each a linear classifier trained separately per layer (16
layers) on the last-token bar activations from Step 8, fit on the train split
and evaluated on the test split:

    mode      2 classes   major / minor
    key      ~18 classes  the full key string, e.g. 'B- major'
    relative  ~9 classes  keys sharing a key signature collapsed:
                          minor -> its RELATIVE major (A minor -> C major)
    parallel  ~9 classes  keys sharing a tonal center collapsed:
                          minor -> its PARALLEL major (C minor -> C major)

All labels derive from the per-split keys.csv written by Step 7. Test samples
whose label never occurs in train are dropped (reported). Class counts are
whatever the data yields — they are printed, not assumed.

Output (under --output):
    probe_results.json   per task: classes, sample counts, per-layer test accuracy
    probe_weights.pt     per task: W [n_layers, C, d_model], b [n_layers, C]
                         (kept for the later steering experiments)

CPU-only. Run after Step 8.

Usage:
    python reproduce/train_probes.py
    python reproduce/train_probes.py --tasks mode key
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

N_LAYERS = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Minor tonic -> relative major tonic (shared key signature), in music21
# pitch-name spelling. Covers enharmonic spellings; unmapped keys raise.
MINOR_TO_RELATIVE_MAJOR = {
    "A": "C",  "E": "G",  "B": "D",  "F#": "A",  "C#": "E",  "G#": "B",
    "D": "F",  "G": "B-", "C": "E-", "F": "A-",  "B-": "D-", "E-": "G-",
    "A-": "C-", "D#": "F#", "A#": "C#",
}


def derive_label(task, key_str):
    """Map a 'Tonic mode' key string to the task's label."""
    tonic, mode = key_str.rsplit(" ", 1)
    if task == "mode":
        return mode
    if task == "key":
        return key_str
    if mode == "major":
        return key_str
    if task == "relative":
        if tonic not in MINOR_TO_RELATIVE_MAJOR:
            raise ValueError(f"no relative-major mapping for tonic {tonic!r}")
        return f"{MINOR_TO_RELATIVE_MAJOR[tonic]} major"
    if task == "parallel":
        return f"{tonic} major"
    raise ValueError(f"unknown task {task!r}")


def load_split(probes_data, split):
    """Return (X [N, n_layers, d_model] float32 tensor, key_strs list of N)."""
    split_dir = os.path.join(probes_data, split)
    with open(os.path.join(split_dir, "keys.csv")) as f:
        keys = {row["chorale"]: row["key"] for row in csv.DictReader(f)}

    feats, labels = [], []
    for chorale in sorted(keys):
        blob = torch.load(os.path.join(split_dir, chorale, "activations.pt"),
                          map_location="cpu", weights_only=False)
        feats.append(blob["activations"])
        labels.extend([keys[chorale]] * blob["activations"].shape[0])
    return torch.cat(feats), labels


def fit_logreg(X, y, n_classes, l2=1.0, max_iter=500):
    """Multinomial logistic regression, full-batch LBFGS.

    Same objective as sklearn's default LogisticRegression (C=1.0):
    sum of cross-entropies + 0.5 * l2 * ||W||^2. Pure torch — sklearn's
    binaries need a newer glibc than this cluster has.
    """
    W = torch.zeros(n_classes, X.shape[1], device=X.device, requires_grad=True)
    b = torch.zeros(n_classes, device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (torch.nn.functional.cross_entropy(X @ W.T + b, y, reduction="sum")
                + 0.5 * l2 * (W ** 2).sum())
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def train_task(task, X_tr, keys_tr, X_te, keys_te):
    """Fit one probe per layer. Returns (results dict, weights dict)."""
    y_tr = np.array([derive_label(task, k) for k in keys_tr])
    y_te = np.array([derive_label(task, k) for k in keys_te])

    classes = sorted(set(y_tr))
    keep = np.isin(y_te, classes)
    n_dropped = int((~keep).sum())
    X_te, y_te = X_te[keep], y_te[keep]

    cls_idx = {c: i for i, c in enumerate(classes)}
    yi_tr = torch.tensor([cls_idx[c] for c in y_tr], device=DEVICE)
    yi_te = torch.tensor([cls_idx[c] for c in y_te], device=DEVICE)

    print(f"\n=== {task}: {len(classes)} classes, "
          f"{len(y_tr)} train, {len(y_te)} test ({n_dropped} test dropped) ===")

    accs, Ws, bs = [], [], []
    for layer in range(N_LAYERS):
        W, b = fit_logreg(X_tr[:, layer].to(DEVICE), yi_tr, len(classes))
        pred = (X_te[:, layer].to(DEVICE) @ W.T + b).argmax(dim=1)
        acc = (pred == yi_te).float().mean().item()
        accs.append(acc)
        Ws.append(W.cpu())
        bs.append(b.cpu())
        print(f"    layer {layer:2d}: acc={acc:.3f}")

    results = {"classes": classes, "n_train": len(y_tr), "n_test": len(y_te),
               "n_test_dropped": n_dropped, "accuracy": accs}
    weights = {"classes": classes, "W": torch.stack(Ws), "b": torch.stack(bs)}
    return results, weights


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probes_data", default="reproduce_output/probes_data")
    p.add_argument("--output", default="reproduce_output/probes")
    p.add_argument("--tasks", nargs="+", default=["mode", "key", "relative", "parallel"],
                   choices=["mode", "key", "relative", "parallel"])
    args = p.parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("Loading activations ...")
    X_tr, keys_tr = load_split(args.probes_data, "train")
    X_te, keys_te = load_split(args.probes_data, "test")
    print(f"train: {tuple(X_tr.shape)}   test: {tuple(X_te.shape)}")

    all_results, all_weights = {}, {}
    for task in args.tasks:
        all_results[task], all_weights[task] = train_task(
            task, X_tr, keys_tr, X_te, keys_te)

    res_path = os.path.join(args.output, "probe_results.json")
    with open(res_path, "w") as f:
        json.dump(all_results, f, indent=2)
    w_path = os.path.join(args.output, "probe_weights.pt")
    torch.save(all_weights, w_path)
    print(f"\nResults -> {res_path}\nWeights -> {w_path}")


if __name__ == "__main__":
    main()

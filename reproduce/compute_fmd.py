#!/usr/bin/env python3
"""
Compute Frechet Music Distance (FMD) for the steering experiments — the
generation-quality metric on the steered continuations.

Scores each generated set against the Bach reference — lower = closer to real Bach —
using the frechet-music-distance library (Retkowski, Stepniak & Modrzejewski, 2024,
arXiv:2412.07948) with the CLaMP 2 feature extractor:

    FMD(bach, aria)       generation quality of the UNSTEERED model
    FMD(bach, mode)       "            "        under mode steering
    FMD(bach, relative)   "            "        under relative-minor steering
    FMD(bach, parallel)   "            "        under parallel-minor steering

Reads   reproduce_output/fmd_cont_data/{bach,aria,mode,relative,parallel}/  (Step 14)
Writes  reproduce_output/fmd_cont_data/fmd_cont_scores.json  and prints a table.

The default sets are the CONTINUATION-ONLY windows (build_fmd_cont_data.py): one bar of
generated (or real-Bach) material after the cut, nothing before it — this is the FMD
variant reported in the paper's Table 1. Pass --fmd_data reproduce_output/fmd_data to
score the 2-bar centred windows instead (those share a bar of real-Bach context across sets).

REQUIRES A GPU and the library:
    pip install frechet-music-distance
The CLaMP 2 model (~a few GB) downloads automatically on the first run.

Note on sample size: each set has ~136 pieces, small for a Frechet metric, so the
ABSOLUTE FMD carries a small-sample (upward) bias. But every condition shares the same
n and the same reference, so that bias is common to all — the RELATIVE ordering
(aria vs each steered condition) is the meaningful read. `--inf` additionally reports
FMD-Inf, which extrapolates the bias away.

Usage:
    python reproduce/compute_fmd.py
    python reproduce/compute_fmd.py --model clamp          # CLaMP instead of CLaMP 2
    python reproduce/compute_fmd.py --inf                  # also report FMD-Inf
    python reproduce/compute_fmd.py --estimator bootstrap  # robust covariance estimator
"""

import argparse
import json
import os


def n_mid(d):
    return len([f for f in os.listdir(d) if f.endswith(".mid")]) if os.path.isdir(d) else 0


def _inf_value(res):
    """Pull the scalar score out of whatever score_inf returns (namedtuple/dict/float)."""
    for attr in ("score", "fmd_inf", "value"):
        if hasattr(res, attr):
            return float(getattr(res, attr))
    if isinstance(res, dict):
        for k in ("score", "fmd_inf", "value"):
            if k in res:
                return float(res[k])
    try:
        return float(res)
    except (TypeError, ValueError):
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fmd_data", default="reproduce_output/fmd_cont_data")
    p.add_argument("--reference", default="bach")
    p.add_argument("--conditions", nargs="+",
                   default=["aria", "mode", "relative", "parallel"])
    p.add_argument("--model", default="clamp2", choices=["clamp2", "clamp"])
    p.add_argument("--estimator", default=None,
                   help="gaussian_estimator for the covariance (library default if unset; "
                        "a shrinkage/bootstrap estimator is steadier at small n)")
    p.add_argument("--inf", action="store_true",
                   help="also compute FMD-Inf (sample-size-corrected extrapolation)")
    p.add_argument("--inf_min_n", type=int, default=30)
    p.add_argument("--out", default="reproduce_output/fmd_cont_data/fmd_cont_scores.json")
    args = p.parse_args()

    from frechet_music_distance import FrechetMusicDistance

    ref_dir = os.path.join(args.fmd_data, args.reference)
    if not os.path.isdir(ref_dir):
        raise SystemExit(f"reference dir not found: {ref_dir} (run Step 14 first)")
    n_ref = n_mid(ref_dir)
    print(f"reference: {args.reference} ({n_ref} files)   model: {args.model}"
          f"{'  estimator: ' + args.estimator if args.estimator else ''}\n")

    kw = {"feature_extractor": args.model, "verbose": True}
    if args.estimator:
        kw["gaussian_estimator"] = args.estimator
    metric = FrechetMusicDistance(**kw)

    results = {}
    for cond in args.conditions:
        test_dir = os.path.join(args.fmd_data, cond)
        n = n_mid(test_dir)
        if n == 0:
            print(f"  {cond}: no MIDIs, skipped")
            continue
        # Bach features are cached after the first call, so they are reused across conditions.
        score = float(metric.score(reference_path=ref_dir, test_path=test_dir))
        entry = {"fmd": score, "n_test": n}
        if args.inf:
            inf = metric.score_inf(reference_path=ref_dir, test_path=test_dir,
                                   min_n=args.inf_min_n)
            entry["fmd_inf"] = _inf_value(inf)
        results[cond] = entry
        extra = f"   FMD-Inf={entry.get('fmd_inf'):.4f}" if args.inf and entry.get("fmd_inf") is not None else ""
        print(f"  FMD(bach, {cond:9s}) = {score:8.4f}   (n={n}){extra}")

    print("\n=== FMD vs Bach (lower = closer to real Bach) ===")
    header = f"{'condition':10} {'n':>4} {'FMD':>9}" + ("   FMD-Inf" if args.inf else "")
    print(header)
    print("-" * len(header))
    for cond, e in results.items():
        line = f"{cond:10} {e['n_test']:>4} {e['fmd']:>9.4f}"
        if args.inf and e.get("fmd_inf") is not None:
            line += f" {e['fmd_inf']:>9.4f}"
        print(line)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"reference": args.reference, "n_reference": n_ref,
                   "model": args.model, "estimator": args.estimator,
                   "scores": results}, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()

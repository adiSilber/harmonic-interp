#!/usr/bin/env python3
"""
Prepare the steering dataset: V->I cadence cuts of the MAJOR chorales in the
test and eval splits (paper Section 5.3 / Figure 5).

Steering focuses on major chorales, where a V->I cadence is the natural case for
a harmonic modification. This reuses the Step-2 cadence pipeline (analyze + cut)
but on the test/valid splits, and WITHOUT minorization: steering adds
probe-derived direction vectors to the residual stream, it does not use the
minorized counterfactuals that the patching experiment needed.

For each major chorale it writes, under <output>/<split>/:
    chorale_XXXX_chords.txt            harmonic analysis + detected cadences
    chorale_XXXX_cuts_output/          one MIDI per V->I cut point
                                       (chorale truncated at the end of the V chord)

Non-major chorales are skipped by the analysis step (wrong mode).

CPU-only. Run after Step 1 (download_jsb_chorales.py).

Usage:
    python reproduce/prepare_steering_data.py
    python reproduce/prepare_steering_data.py --splits test
"""

import argparse
import os
import sys

# Ensure this reproduce/ dir is on sys.path so the sibling module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prepare_chorale_data import run_analyze, run_cut

# steering split name -> source dir name (Step 1 output)
SPLIT_MAP = {"test": "test_16th", "eval": "valid_16th"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="reproduce_output/jsb_chorales_midi",
                   help="Step 1 output dir with {test,valid}_16th/")
    p.add_argument("--output", default="reproduce_output/steering_data")
    p.add_argument("--splits", nargs="+", default=["test", "eval"],
                   choices=["test", "eval"])
    args = p.parse_args()

    for split in args.splits:
        src_dir = os.path.join(args.input, SPLIT_MAP[split])
        out_dir = os.path.join(args.output, split)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n=== {split}: major V->I cadence cuts ===")
        run_analyze(src_dir, out_dir, target_mode="major")
        run_cut(src_dir, out_dir, out_dir)


if __name__ == "__main__":
    main()

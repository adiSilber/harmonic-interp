#!/usr/bin/env python3
"""
Label the resolution roman numeral of every steered / baseline continuation (Figure 5).

For each output.mid from Step 12-15, re-tokenize it, split off the generated tokens
(everything after the prompt = the Step-10 cut it was generated from), take the first
continuation chord, and label it with a roman numeral relative to the chorale's
original MAJOR key (the same "resulting harmony" logic used for Figures 2/3).

Output: reproduce_output/steering/resolutions.csv
    columns: condition, split, chorale, cut, rn

CPU-only (tokenizer + music21, no GPU / no torch). Run after Steps 12-15.

Usage:
    python reproduce/label_resolutions.py
    python reproduce/label_resolutions.py --conditions baseline mode
"""

import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ariautils.tokenizer import AbsTokenizer
from utils import resulting_harmony_rn, _parse_key_from_chords_txt


def tokenize_midi(tokenizer, path):
    from ariautils.midi import MidiDict
    md = MidiDict.from_midi(path)
    return tokenizer.encode(tokenizer.tokenize(md, add_dim_tok=False, add_eos_tok=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steering", default="reproduce_output/steering")
    p.add_argument("--steering_data", default="reproduce_output/steering_data")
    p.add_argument("--conditions", nargs="+",
                   default=["baseline", "mode", "relative", "parallel"])
    p.add_argument("--out", default="reproduce_output/steering/resolutions.csv")
    args = p.parse_args()

    tok = AbsTokenizer()
    key_cache, plen_cache = {}, {}
    rows = []

    for cond in args.conditions:
        outs = sorted(glob.glob(os.path.join(args.steering, cond,
                                             "*", "*", "*", "output.mid")))
        n_ok = n_none = 0
        for out in outs:
            # .../<cond>/<split>/<chorale>/<cut_stem>/output.mid
            cut_stem = os.path.basename(os.path.dirname(out))
            chorale = os.path.basename(os.path.dirname(os.path.dirname(out)))
            split = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(out))))

            chords = os.path.join(args.steering_data, split, f"{chorale}_chords.txt")
            cut_midi = os.path.join(args.steering_data, split,
                                    f"{chorale}_cuts_output", f"{cut_stem}.mid")
            if chords not in key_cache:
                key_cache[chords] = _parse_key_from_chords_txt(chords)
            key = key_cache[chords]
            if key is None or not os.path.exists(cut_midi):
                continue
            if cut_midi not in plen_cache:
                plen_cache[cut_midi] = len(tokenize_midi(tok, cut_midi))
            prompt_len = plen_cache[cut_midi]

            out_ids = tokenize_midi(tok, out)
            n_gen = len(out_ids) - prompt_len
            if n_gen <= 0:
                continue
            meta = {"capture_positions": list(range(len(out_ids))), "token_ids": out_ids}
            rn = resulting_harmony_rn(meta, key, tok, n_gen)
            rows.append((cond, split, chorale, cut_stem, rn or ""))
            n_ok += rn is not None
            n_none += rn is None
        print(f"{cond:9s}: {len(outs)} outputs, {n_ok} labelled, {n_none} no-chord")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "split", "chorale", "cut", "rn"])
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()

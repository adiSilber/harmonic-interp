#!/usr/bin/env python3
"""
Prepare the probing dataset: bar-truncated MIDIs + a per-split key table.

For the train and test splits (the probe is fit on train and evaluated on test;
the valid split is not used), this writes:
    <output>/<split>/keys.csv                                one row per chorale: chorale,key
    <output>/<split>/chorale_XXXX/chorale_XXXX_bars_1-N.mid  chorale truncated at bar N's end

Bars. Everything in these files sits on a 16th-note grid: ticks_per_beat is 480,
but the gcd of all note event times is 120 ticks = one 16th, so the 16th is the
real unit. A 4/4 bar is 16 of them. We cut at every boundary k*16 while at least
one full bar remains after the cut, so the trailing partial bar is never emitted.

Caveat: a minority of chorales contain an irregular measure (12 sixteenths
instead of 16), so for those the fixed-16 grid drifts from the notated barline.
Measured against the reference dataset: 17/213 train and 3/66 test chorales.
No chorale is skipped on that account.

Labels. The key ("<tonic> <mode>", e.g. "B- major" / "A minor") is all the probe
tasks in the paper need (mode, key, tonic). It is written once per split as a
keys.csv table rather than a text file per chorale.

CPU-only — no GPU needed. Run after Step 1 (download_jsb_chorales.py).

Usage:
    python reproduce/prepare_probe_data.py
    python reproduce/prepare_probe_data.py --splits train
    python reproduce/prepare_probe_data.py --output reproduce_output/probes_data
"""

import argparse
import csv
import os
import sys

import mido
import music21

# Ensure this reproduce/ dir is on sys.path so the sibling modules import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prepare_chorale_data import cut_midi_at_16th

TICKS_PER_16TH = 120
SIXTEENTHS_PER_BAR = 16          # one 4/4 bar

# probes split name -> source dir name (Step 1 output)
SPLIT_MAP = {"train": "train_16th", "test": "test_16th"}


def midi_length_16ths(path):
    """Total length of a MIDI file in 16th notes."""
    m = mido.MidiFile(path)
    ticks = max(sum(msg.time for msg in track) for track in m.tracks)
    return ticks // TICKS_PER_16TH


def bar_cut_16ths(total_16ths):
    """Bar-end cut points, in 16ths, while >= 1 full bar remains after the cut."""
    return [k * SIXTEENTHS_PER_BAR
            for k in range(1, total_16ths // SIXTEENTHS_PER_BAR + 1)
            if total_16ths - k * SIXTEENTHS_PER_BAR >= SIXTEENTHS_PER_BAR]


def chorale_key(midi_path):
    """Key of a chorale, e.g. 'B- major' / 'A minor'.

    Built from tonic.name rather than str(key): music21 renders minor keys with
    a lower-case tonic ('a minor'), while the probe's class list expects the
    pitch-name spelling ('A minor').
    """
    k = music21.converter.parse(midi_path).analyze('key')
    return f"{k.tonic.name} {k.mode}"


def cut_bars(num, src_midi, out_split_dir):
    """Write the bar-truncated MIDIs for one chorale. Returns the bar count."""
    chorale = f"chorale_{num:04d}"
    cuts = bar_cut_16ths(midi_length_16ths(src_midi))
    chorale_dir = os.path.join(out_split_dir, chorale)
    os.makedirs(chorale_dir, exist_ok=True)
    for bar_num, cut_16th in enumerate(cuts, start=1):
        out_midi = os.path.join(chorale_dir, f"{chorale}_bars_1-{bar_num}.mid")
        if not os.path.exists(out_midi):
            cut_midi_at_16th(src_midi, cut_16th, out_midi)
    return len(cuts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="reproduce_output/jsb_chorales_midi",
                   help="Step 1 output dir with {train,test}_16th/")
    p.add_argument("--output", default="reproduce_output/probes_data")
    p.add_argument("--splits", nargs="+", default=["train", "test"],
                   choices=["train", "test"])
    args = p.parse_args()

    for split in args.splits:
        src_dir = os.path.join(args.input, SPLIT_MAP[split])
        out_split_dir = os.path.join(args.output, split)
        os.makedirs(out_split_dir, exist_ok=True)

        midis = sorted(f for f in os.listdir(src_dir)
                       if f.startswith("chorale_") and f.endswith(".mid"))
        print(f"\n=== {split}: {len(midis)} chorales ===")
        rows, total_bars = [], 0
        for i, fname in enumerate(midis, start=1):
            num = int(fname[len("chorale_"):-len(".mid")])
            src_midi = os.path.join(src_dir, fname)
            total_bars += cut_bars(num, src_midi, out_split_dir)
            rows.append((f"chorale_{num:04d}", chorale_key(src_midi)))
            if i % 25 == 0 or i == len(midis):
                print(f"    {i}/{len(midis)} chorales, {total_bars} bar files")

        keys_path = os.path.join(out_split_dir, "keys.csv")
        with open(keys_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["chorale", "key"])
            w.writerows(rows)
        print(f"    done: {len(midis)} chorales -> {total_bars} bar MIDIs, keys -> {keys_path}")


if __name__ == "__main__":
    main()

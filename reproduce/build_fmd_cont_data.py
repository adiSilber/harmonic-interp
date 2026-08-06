#!/usr/bin/env python3
"""
Build CONTINUATION-ONLY MIDI sets for FMD — the variant reported in the paper.

`build_fmd_data.py` writes a 32-sixteenth window CENTRED on the cut: 16 sixteenths of
real-Bach context before it and 16 after. That context bar is byte-identical in bach/ and
in every generated set, so half of every clip FMD sees is shared across sets.

This script keeps ONLY the bar after the cut — what each set actually produced:

    bach/      the real Bach notes that follow the cut
    aria/      Aria's UNSTEERED continuation
    mode/      Aria's mode-steered continuation
    relative/  Aria's relative-minor-steered continuation
    parallel/  Aria's parallel-minor-steered continuation

The last prompt chord sits exactly at cut_ms and the left bound is strict, so that shared
chord drops out too and nothing but generated material remains.

Everything else matches build_fmd_data.py: same cut points, same 1:1 file naming, releases
clipped to the right edge, shifted to t=0, tempo 500000 / 480 ppq. Writes to a SEPARATE
tree so the existing 32-sixteenth sets and their scores are untouched.

Output: reproduce_output/fmd_cont_data/<set>/<split>_<chorale>_tick<T>.mid
CPU-only (tokenizer, no torch).

WARNING: frechet-music-distance disk-caches extracted features (joblib.Memory under
~/.cache/frechet_music_distance/precomputed) keyed on the DIRECTORY PATH alone. Rebuilding
a set in place and rescoring silently returns the OLD features. After any rebuild run:
    python -c "from frechet_music_distance.utils import clear_cache; clear_cache()"
or write to a fresh output directory.

Usage:
    python reproduce/build_fmd_cont_data.py
    python reproduce/build_fmd_cont_data.py --steering reproduce_output/steering_l12 \
        --conditions relative parallel --set_suffix _l12
    python reproduce/compute_fmd.py --fmd_data reproduce_output/fmd_cont_data \
        --conditions aria mode relative parallel \
        --out reproduce_output/fmd_cont_data/fmd_cont_scores.json
"""

import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ariautils.tokenizer import AbsTokenizer
from ariautils.midi import MidiDict

from build_fmd_data import COND_TO_SET, SPLIT_MAP, MS_PER_16TH, TICKS_PER_BEAT, MS_PER_BEAT
from utils import decode_tokens_to_notes

import mido

N_16TH_AFTER = 16          # one bar of continuation; nothing before the cut


def continuation_notes(notes, cut_ms, after_ms):
    """Everything SOUNDING in (cut_ms, cut_ms + after_ms), clipped to that span.

    Selection is by overlap, not by onset. `build_fmd_data.window_notes` keeps a note only
    if its ONSET falls inside the window, which silently deletes any voice that was struck
    earlier and is still held — at a cadence that is usually the bass and tenor, so the clip
    ends up a chord with its lower voices missing. Here a note is kept whenever its sounding
    span intersects the window, with its onset pulled up to the left edge and its release
    clipped to the right one.

    The left bound is exclusive of notes that merely END at cut_ms, and a note struck exactly
    at cut_ms is the shared final prompt chord — held ones survive as sustain, which is what
    is actually sounding, while re-articulations after the cut come through as new onsets."""
    right = cut_ms + after_ms
    out = []
    for n in notes:
        start, end = n["onset"], n["onset"] + n["duration"]
        if end <= cut_ms or start >= right:
            continue
        m = dict(n)
        m["onset"] = max(start, cut_ms)
        m["duration"] = min(end, right) - m["onset"]
        if m["duration"] > 0:
            out.append(m)
    return out


def write_window_midi(notes, out_path, origin_ms):
    """Write notes to MIDI with t=0 pinned at origin_ms. False if empty.

    Unlike `build_fmd_data.notes_to_midi`, the time origin is the window's left edge rather
    than the first onset, so a clip whose first note arrives late keeps its leading rest and
    every set stays on the same time reference."""
    if not notes:
        return False

    def tick(ms):
        return max(0, int(round((ms - origin_ms) * TICKS_PER_BEAT / MS_PER_BEAT)))

    events = []
    for n in notes:
        on = tick(n["onset"])
        off = tick(n["onset"] + n["duration"])
        if off <= on:
            off = on + 1
        events.append((on, 1, n["midi"], n["velocity"]))
        events.append((off, 0, n["midi"], 0))
    events.sort(key=lambda e: (e[0], e[1]))

    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    prev = 0
    for t, typ, note, vel in events:
        tr.append(mido.Message("note_on" if typ else "note_off",
                               note=note, velocity=vel, time=t - prev))
        prev = t
    tr.append(mido.MetaMessage("end_of_track", time=0))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mid.save(out_path)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steering", default="reproduce_output/steering")
    p.add_argument("--steering_data", default="reproduce_output/steering_data")
    p.add_argument("--jsb", default="reproduce_output/jsb_chorales_midi")
    p.add_argument("--out", default="reproduce_output/fmd_cont_data")
    p.add_argument("--conditions", nargs="+",
                   default=["baseline", "mode", "relative", "parallel"])
    p.add_argument("--set_suffix", default="",
                   help="appended to each output set name, e.g. '_l12'")
    p.add_argument("--after", type=int, default=N_16TH_AFTER,
                   help="sixteenths of continuation to keep after the cut (default 16 = 1 bar)")
    p.add_argument("--limit", type=int, default=None, help="max cuts per condition (smoke)")
    args = p.parse_args()

    after_ms = args.after * MS_PER_16TH
    tok = AbsTokenizer()
    cache = {}
    prompt_end_cache = {}
    bach_done = set()

    def tok_notes(midi_path):
        if midi_path not in cache:
            md = MidiDict.from_midi(midi_path)
            ids = tok.encode(tok.tokenize(md, add_dim_tok=False, add_eos_tok=False))
            cache[midi_path] = decode_tokens_to_notes(ids, list(range(len(ids))), tok)
        return cache[midi_path]

    def prompt_end_ms(cut_midi):
        if cut_midi not in prompt_end_cache:
            prompt_end_cache[cut_midi] = max(
                (n["onset"] for n in tok_notes(cut_midi)), default=0.0)
        return prompt_end_cache[cut_midi]

    for cond in args.conditions:
        set_name = COND_TO_SET[cond] + args.set_suffix
        outs = sorted(glob.glob(os.path.join(
            args.steering, cond, "*", "*", "*", "output.mid")))
        if args.limit:
            outs = outs[:args.limit]
        n_ok = n_empty = n_bach = 0
        for out in outs:
            cut_stem = os.path.basename(os.path.dirname(out))
            chorale = os.path.basename(os.path.dirname(os.path.dirname(out)))
            split = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(out))))
            m = re.search(r"_cut_tick_(\d+)$", cut_stem)
            tick = int(m.group(1)) if m else None
            base = f"{split}_{chorale}_tick{tick}"

            cut_midi = os.path.join(args.steering_data, split,
                                    f"{chorale}_cuts_output", f"{cut_stem}.mid")
            if not os.path.exists(cut_midi):
                continue
            pe = prompt_end_ms(cut_midi)

            gen = continuation_notes(tok_notes(out), pe, after_ms)
            if write_window_midi(gen, os.path.join(args.out, set_name, base + ".mid"), pe):
                n_ok += 1
            else:
                n_empty += 1

            key = (split, chorale, tick)
            if key not in bach_done:
                orig = os.path.join(args.jsb, SPLIT_MAP[split], f"{chorale}.mid")
                if os.path.exists(orig):
                    if write_window_midi(continuation_notes(tok_notes(orig), pe, after_ms),
                                         os.path.join(args.out, "bach", base + ".mid"), pe):
                        n_bach += 1
                bach_done.add(key)
        print(f"{set_name:9s}: {len(outs)} cuts -> {n_ok} written "
              f"({n_empty} empty), +{n_bach} bach")

    total_bach = len(glob.glob(os.path.join(args.out, "bach", "*.mid")))
    print(f"\nbach reference set: {total_bach} files ({args.after} sixteenths each)")
    print(f"Output -> {args.out}/<set>/<split>_<chorale>_tick<T>.mid")


if __name__ == "__main__":
    main()

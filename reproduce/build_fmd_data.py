#!/usr/bin/env python3
"""
Build the MIDI sets for Frechet Music Distance (FMD) evaluation of the steering
experiments — the generation-quality metric for the steered continuations.

  Retkowski, Stepniak & Modrzejewski (2024), "Frechet Music Distance", arXiv:2412.07948.

For each V->I cut we take a window CENTERED on the cut — 16 sixteenths (one bar) before
it and 16 sixteenths after — and emit one short MIDI per set, all shifted to start at
t=0. The "before" bar is identical real-Bach context in every set; the "after" bar is
where Bach / Aria / the steered model diverge:

    bach/      the real Bach notes that follow the cut (from the original chorale)
    aria/      Aria's UNSTEERED continuation (the steering `baseline` output)
    mode/      Aria's mode-steered continuation
    relative/  Aria's relative-minor-steered continuation
    parallel/  Aria's parallel-minor-steered continuation

FMD is then computed set-vs-set against `bach/` (lower = closer to real Bach). Files
are named <split>_<chorale>_tick<T>.mid so the sets line up 1:1.

The window is symmetric across every set: notes with
    cut_ms - 16*125ms < onset <= cut_ms + 16*125ms
where cut_ms is the onset of the last prompt chord (from the cut MIDI). This keeps
Bach and every generated set the same length and time-reference per cut.

Output: reproduce_output/fmd_data/<set>/<split>_<chorale>_tick<T>.mid
CPU-only (tokenizer + music21, no torch).

Usage:
    python reproduce/build_fmd_data.py                 # all sets, test+eval
    python reproduce/build_fmd_data.py --limit 3       # smoke test
"""

import argparse
import glob
import os
import re
import sys

import mido

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ariautils.tokenizer import AbsTokenizer
from ariautils.midi import MidiDict
from utils import decode_tokens_to_notes

SPLIT_MAP = {"test": "test_16th", "eval": "valid_16th"}
# steering condition dir -> FMD set name
COND_TO_SET = {"baseline": "aria", "mode": "mode", "relative": "relative", "parallel": "parallel"}

TICKS_PER_BEAT = 480
MS_PER_BEAT = 500.0   # 120 BPM (tempo 500000 us/beat), matches download_jsb_chorales.py
MS_PER_16TH = 125.0   # one sixteenth at 120 BPM
N_16TH_BEFORE = 16    # one bar of real-Bach context before the cut
N_16TH_AFTER = 16     # one bar of continuation after the cut
BEFORE_MS = N_16TH_BEFORE * MS_PER_16TH
AFTER_MS = N_16TH_AFTER * MS_PER_16TH


def tok_notes(tok, midi_path, cache):
    """Tokenize a MIDI and decode to note dicts (cached by path)."""
    if midi_path in cache:
        return cache[midi_path]
    md = MidiDict.from_midi(midi_path)
    ids = tok.encode(tok.tokenize(md, add_dim_tok=False, add_eos_tok=False))
    notes = decode_tokens_to_notes(ids, list(range(len(ids))), tok)
    cache[midi_path] = (ids, notes)
    return ids, notes


def window_notes(notes, cut_ms):
    """Symmetric window around the cut: onset in (cut_ms - BEFORE_MS, cut_ms + AFTER_MS)
    = 16 sixteenths of real-Bach context before + 16 sixteenths after the cut. Each note's
    release is clipped to the right edge so no held note runs past the 32-sixteenth window."""
    left, right = cut_ms - BEFORE_MS, cut_ms + AFTER_MS
    out = []
    for n in notes:
        if left < n["onset"] < right:
            m = dict(n)
            m["duration"] = min(n["onset"] + n["duration"], right) - n["onset"]
            out.append(m)
    return out


def notes_to_midi(notes, out_path):
    """Write notes (shifted so the earliest onset is t=0) to a MIDI. False if empty."""
    if not notes:
        return False
    t0 = min(n["onset"] for n in notes)

    def tick(ms):
        return max(0, int(round((ms - t0) * TICKS_PER_BEAT / MS_PER_BEAT)))

    events = []
    for n in notes:
        on = tick(n["onset"])
        off = tick(n["onset"] + n["duration"])
        if off <= on:
            off = on + 1
        events.append((on, 1, n["midi"], n["velocity"]))   # note_on
        events.append((off, 0, n["midi"], 0))              # note_off
    # at equal tick, emit note_off (0) before note_on (1)
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
    p.add_argument("--out", default="reproduce_output/fmd_data")
    p.add_argument("--conditions", nargs="+",
                   default=["baseline", "mode", "relative", "parallel"])
    p.add_argument("--set_suffix", default="",
                   help="appended to each output set name, e.g. '_l12' for the "
                        "single-layer variant of a condition. Keeps the variant beside "
                        "the main sets so it is scored against the same bach/ reference.")
    p.add_argument("--limit", type=int, default=None, help="max cuts per condition (smoke)")
    args = p.parse_args()

    tok = AbsTokenizer()
    cache = {}
    prompt_end_cache = {}   # cut_midi -> prompt_end_ms
    bach_done = set()       # (split, chorale, tick) already written for bach

    def prompt_end_ms(cut_midi):
        if cut_midi not in prompt_end_cache:
            _, pnotes = tok_notes(tok, cut_midi, cache)
            prompt_end_cache[cut_midi] = max((n["onset"] for n in pnotes), default=0.0)
        return prompt_end_cache[cut_midi]

    counts = {}
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

            # generated continuation from this output.mid
            _, out_notes = tok_notes(tok, out, cache)
            gen = window_notes(out_notes, pe)
            if notes_to_midi(gen, os.path.join(args.out, set_name, base + ".mid")):
                n_ok += 1
            else:
                n_empty += 1

            # real Bach continuation from the original chorale (write once)
            key = (split, chorale, tick)
            if key not in bach_done:
                orig = os.path.join(args.jsb, SPLIT_MAP[split], f"{chorale}.mid")
                if os.path.exists(orig):
                    _, onotes = tok_notes(tok, orig, cache)
                    if notes_to_midi(window_notes(onotes, pe),
                                     os.path.join(args.out, "bach", base + ".mid")):
                        n_bach += 1
                bach_done.add(key)
        counts[set_name] = (len(outs), n_ok, n_empty, n_bach)
        print(f"{set_name:9s}: {len(outs)} cuts -> {n_ok} written "
              f"({n_empty} empty), +{n_bach} bach")

    total_bach = len(glob.glob(os.path.join(args.out, "bach", "*.mid")))
    print(f"\nbach reference set: {total_bach} files")
    print(f"Output -> {args.out}/<set>/<split>_<chorale>_tick<T>.mid")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build an ART OF FUGUE contrast set for FMD — real Bach, four voices, different genre.

A contrast set that varies GENRE while holding composer, texture and voice count fixed:
Contrapunctus 1-11 of Die Kunst der Fuge (BWV 1080) is securely Bach and four-voice like
the chorales, so only the musical idiom changes.

    FMD(bach, artfugue)   chorale cadence vs instrumental fugue, same composer

Source: https://github.com/craigsapp/art-of-the-fugue — Humdrum **kern encoded by Craig
Stuart Sapp (CCARH/Stanford) from the Bach-Gesellschaft Ausgabe, Band 25.1 (Leipzig:
Breitkopf und Härtel, 1878), ed. Wilhelm Rust. Each file records its own source edition,
encoder and date, so the provenance is citable — unlike sequenced MIDI collections.

Windows match the FMD reference format exactly: N sixteenths of music, quantized to the
sixteenth grid, one notated quarter = 500ms (120 BPM), 480 ppq, tempo 500000, velocity 60,
shifted so the window's left edge is t=0. Windows are non-overlapping and drawn round-robin
across the eleven fugues so no single piece dominates.

    --sixteenths 32   matches reproduce_output/fmd_data/bach        (centred windows)
    --sixteenths 16   matches reproduce_output/fmd_cont_data/bach   (continuation windows)

Note on tempo: the fugues are notated alla breve, so mapping a notated quarter to 500ms is
a format convention that puts them on the chorales' grid, not a claim about performance
tempo. It is the same convention the JSB pipeline applies to the chorales.

WARNING: frechet-music-distance disk-caches features by directory path. After rebuilding a
set in place, run clear_cache() or the score is silently stale.

Usage:
    python reproduce/build_fmd_artfugue_data.py
    python reproduce/build_fmd_artfugue_data.py --sixteenths 16 \
        --out reproduce_output/fmd_cont_data/artfugue
"""

import argparse
import os
import random
import urllib.request

import mido

KERN_URL = ("https://raw.githubusercontent.com/craigsapp/art-of-the-fugue/master/kern/"
            "artfugue-{:03d}.krn")
CONTRAPUNCTUS = range(1, 12)        # Contrapunctus 1-11 = files 001-011, all four-voice

TICKS_PER_BEAT = 480
MS_PER_BEAT = 500.0                 # one notated quarter, 120 BPM — matches the JSB pipeline
MS_PER_16TH = 125.0
VELOCITY = 60                       # matches the tokenizer output in the reference sets


def fetch_kern(kern_dir):
    """Download Contrapunctus 1-11 (cached). Returns list of local paths."""
    os.makedirs(kern_dir, exist_ok=True)
    paths = []
    for n in CONTRAPUNCTUS:
        p = os.path.join(kern_dir, f"artfugue-{n:03d}.krn")
        if not os.path.exists(p):
            urllib.request.urlretrieve(KERN_URL.format(n), p)
            print(f"  downloaded {os.path.basename(p)}")
        paths.append(p)
    return paths


def kern_notes(path):
    """Parse a **kern score -> [(onset_ms, dur_ms, midi)], quantized to the 16th grid.

    Returns (notes, n_quantized) where n_quantized counts events whose notated position or
    length was not already a multiple of a sixteenth (32nds, triplets) and had to be moved."""
    from music21 import converter

    score = converter.parse(path).stripTies()
    notes, moved = [], 0
    for el in score.flatten().notes:
        on_q, len_q = float(el.offset), float(el.quarterLength)
        if len_q <= 0:
            continue
        on_ms, dur_ms = on_q * MS_PER_BEAT, len_q * MS_PER_BEAT
        q_on = round(on_ms / MS_PER_16TH) * MS_PER_16TH
        q_dur = max(1, round(dur_ms / MS_PER_16TH)) * MS_PER_16TH
        if abs(q_on - on_ms) > 1e-6 or abs(q_dur - dur_ms) > 1e-6:
            moved += 1
        for p in (el.pitches if el.isChord else [el.pitch]):
            notes.append((q_on, q_dur, int(p.midi)))
    notes.sort()
    return notes, moved


def windows(notes, win_ms, min_notes):
    """Non-overlapping [start, start+win_ms) windows, clipped, keyed by start time.

    Selection is by OVERLAP, so a voice struck before the window and still sounding is kept
    and clipped — the same rule the fixed continuation builder uses, and the one the original
    onset-only windowing got wrong."""
    if not notes:
        return []
    last = max(o + d for o, d, _ in notes)
    out = []
    start = 0.0
    while start + win_ms <= last:
        w = []
        for o, d, p in notes:
            if o + d <= start or o >= start + win_ms:
                continue
            on = max(o, start)
            off = min(o + d, start + win_ms)
            if off > on:
                w.append((on - start, off - on, p))
        if len(w) >= min_notes:
            out.append((start, w))
        start += win_ms
    return out


def write_midi(notes, out_path):
    """notes as (onset_ms, dur_ms, midi) already relative to the window start."""
    events = []
    for on_ms, dur_ms, p in notes:
        on = max(0, int(round(on_ms * TICKS_PER_BEAT / MS_PER_BEAT)))
        off = max(0, int(round((on_ms + dur_ms) * TICKS_PER_BEAT / MS_PER_BEAT)))
        if off <= on:
            off = on + 1
        events.append((on, 1, p, VELOCITY))
        events.append((off, 0, p, 0))
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kern_dir", default="reproduce_output/artfugue_kern")
    p.add_argument("--out", default="reproduce_output/fmd_data/artfugue")
    p.add_argument("--sixteenths", type=int, default=32,
                   help="window length; 32 matches fmd_data/bach, 16 matches fmd_cont_data/bach")
    p.add_argument("--n", type=int, default=136, help="windows to emit (match the reference)")
    p.add_argument("--min_notes", type=int, default=8,
                   help="skip near-empty windows (rests, final fermatas)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    win_ms = args.sixteenths * MS_PER_16TH
    rng = random.Random(args.seed)

    print(f"fetching Contrapunctus 1-11 into {args.kern_dir} ...")
    paths = fetch_kern(args.kern_dir)

    per_piece, total_moved, total_notes = {}, 0, 0
    for path in paths:
        stem = os.path.basename(path)[:-4]
        notes, moved = kern_notes(path)
        total_moved += moved
        total_notes += len(notes)
        w = windows(notes, win_ms, args.min_notes)
        rng.shuffle(w)
        per_piece[stem] = w
        print(f"  {stem}: {len(notes):5d} notes -> {len(w):3d} windows")

    if total_notes:
        print(f"\nquantization to the 16th grid moved {total_moved} events "
              f"({100*total_moved/total_notes:.2f}% of {total_notes})")

    # round-robin so no single fugue dominates the set
    order = sorted(per_piece)
    chosen, depth = [], 0
    while len(chosen) < args.n:
        added = False
        for stem in order:
            if depth < len(per_piece[stem]):
                chosen.append((stem, *per_piece[stem][depth]))
                added = True
                if len(chosen) == args.n:
                    break
        if not added:
            break
        depth += 1
    if len(chosen) < args.n:
        print(f"WARNING: only {len(chosen)} windows available (asked {args.n}); "
              f"FMD is n-sensitive, so rescore the reference at the same n")

    for stem, start, w in chosen:
        tick = int(start / MS_PER_16TH)
        write_midi(w, os.path.join(args.out, f"{stem}_tick{tick}.mid"))

    n_pieces = len({s for s, _, _ in chosen})
    print(f"\nartfugue: {len(chosen)} windows of {args.sixteenths} sixteenths "
          f"from {n_pieces} fugues")
    print(f"Output -> {args.out}/artfugue-NNN_tick<T>.mid")


if __name__ == "__main__":
    main()

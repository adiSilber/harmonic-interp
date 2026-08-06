#!/usr/bin/env python3
"""
Reproduce the paper's Table 1 (Key Steering Evaluation) — generation-quality metrics —
with FMD as the Quality metric.

Per-cell definitions were recovered from the original scripts/generation_quality.py;
the thresholds and aggregations are from the paper's "Evaluating Musical Quality"
paragraph (Sec. 5.3):

  Sequence-level
    struct_error_pct  % of cells whose generated token stream has a malformed
                      <pitch,onset,dur> triple, computed from the raw tokens.json
                      that Step 12 saves alongside each output.mid.
    pitch_repeat_pct  % of cells with a run of >= 4 identical consecutive pitches.
    avg_gen_dur_ms    mean (last gen-note offset - last prompt-note offset).
  Note-level
    out_of_range_pct  % of cells with >= 3 notes outside the piano range (pitch>=88 or <=35).
    note_dur_avg_ms   mean generated-note duration (pooled over all gen notes).
    pitch_var         mean over cells of var(gen-note MIDI pitches).
  Quality
    fmd               FMD(bach, condition), read from fmd_cont_data/fmd_cont_scores.json
                      (the continuation-only windows — the variant reported in the paper).
  Harmony  (over the first 16 sixteenth-slots after the cut)
    rn_entropy_bits   mean over cells of Shannon entropy of the slot-RN distribution.
    top1_dom_pct      mean over cells of (most-common-RN count / 16) * 100.

Reads the full Step-12 generations:
    reproduce_output/steering/<cond>/<split>/<chorale>/<cut>/output.mid
CPU-only (tokenizer + music21 + numpy, no torch).

Usage:
    python reproduce/compute_quality_metrics.py
    python reproduce/compute_quality_metrics.py --keep_silent_slots   # count '-' slots in RN dist
"""

import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ariautils.tokenizer import AbsTokenizer
from ariautils.midi import MidiDict
from utils import decode_tokens_to_notes

import music21.chord as m21chord
import music21.pitch as m21pitch
import music21.roman as m21roman
import music21.key as m21key

# constants recovered from generation_quality.py
PITCH_HIGH = 88
PITCH_LOW = 35
SIXTEENTH_MS = 125.0
REPEAT_RUN = 4          # paper: "runs of >= 4 identical pitches"
OOR_MIN = 3             # paper: ">= 3 notes outside the normal piano range"
N_SLOTS = 16            # first bar
INSTRUMENT_FAMILIES = ("organ", "guitar", "bass", "strings", "ensemble",
                       "brass", "reed", "pipe", "synth_lead")
BAD_SPECIALS = ("<S>", "<P>", "<U>", "<D>")

# steering condition -> FMD set name in fmd_scores.json
COND_TO_FMD = {"baseline": "aria", "mode": "mode", "relative": "relative", "parallel": "parallel"}


def has_structural_error(gen_tokens):
    """Faithful port of generation_quality.walk_fsm: True if the generated token
    stream contains a non-piano-note token (drum / other instrument / prefix / bad
    special) OR a broken note-triple order (must cycle piano -> onset -> dur)."""
    err = False
    state = "NOTE"
    for tok in gen_tokens:
        # bad token (walk_fsm._classify_token_and_count)
        if isinstance(tok, tuple):
            head = tok[0]
            if head == "drum" or head in INSTRUMENT_FAMILIES or head == "prefix":
                err = True
        elif isinstance(tok, str) and tok in BAD_SPECIALS:
            err = True
        # FSM transition (walk_fsm._advance_state)
        if tok == "<E>":
            continue
        head = tok[0] if isinstance(tok, tuple) else None
        if state == "NOTE":
            if head == "piano":
                state = "ONSET"
            elif tok == "<T>":
                pass
            elif head in ("onset", "dur"):
                err = True
        elif state == "ONSET":
            if head == "onset":
                state = "DUR"
            else:
                err = True
        elif state == "DUR":
            if head == "dur":
                state = "NOTE"
            else:
                err = True
    return err


def toks(tok, path):
    return tok.encode(tok.tokenize(MidiDict.from_midi(path), add_dim_tok=False, add_eos_tok=False))


def longest_run(seq):
    best = cur = 0
    prev = object()
    for x in seq:
        cur = cur + 1 if x == prev else 1
        best = max(best, cur)
        prev = x
    return best


def sounding_at(notes, t):
    return [n["midi"] for n in notes if n["onset"] <= t < n["onset"] + n["duration"]]


def slot_rn(midi_pitches, key_obj):
    pcs = sorted(set(midi_pitches))
    if len(pcs) < 2:
        return "–"
    try:
        c = m21chord.Chord([m21pitch.Pitch(midi=p) for p in pcs])
        return m21roman.romanNumeralFromChord(c, key_obj).figure
    except Exception:
        return "?"


def rn_grid(all_notes, prompt_end, key_obj, start=0):
    return [slot_rn(sounding_at(all_notes, prompt_end + (k + 0.5) * SIXTEENTH_MS), key_obj)
            for k in range(start, start + N_SLOTS)]


def shannon_bits(labels):
    c = Counter(labels)
    n = sum(c.values())
    return -sum((v / n) * math.log2(v / n) for v in c.values()) if n else 0.0


def parse_key(chords_path):
    for ln in open(chords_path):
        if ln.startswith("Key:"):
            tonic, mode = ln[len("Key:"):].strip().split()
            return m21key.Key(tonic, mode)
    return None


def analyze_cut(tok, out_mid, cut_mid, key_obj, plen_cache, keep_silent):
    # Prefer the raw token stream (tokens.json) when present: it gives the exact
    # structural-error signal. Fall back to re-tokenizing the MIDI (struct-error
    # then unrecoverable -> None).
    tj = os.path.join(os.path.dirname(out_mid), "tokens.json")
    if os.path.exists(tj):
        d = json.load(open(tj))
        ids, prompt_len = d["token_ids"], d["prompt_len"]
        struct_err = has_structural_error([tok.id_to_tok[t] for t in ids[prompt_len:]])
    else:
        if cut_mid not in plen_cache:
            plen_cache[cut_mid] = len(toks(tok, cut_mid))
        prompt_len = plen_cache[cut_mid]
        ids = toks(tok, out_mid)
        struct_err = None

    all_notes = decode_tokens_to_notes(ids, list(range(len(ids))), tok)
    gen = sorted((n for n in all_notes if n["pitch_position"] >= prompt_len),
                 key=lambda n: n["pitch_position"])
    prompt_end = max((n["onset"] + n["duration"] for n in all_notes
                      if n["pitch_position"] < prompt_len), default=0.0)
    if not gen:
        return None

    pitches = [n["midi"] for n in gen]
    durs = [n["duration"] for n in gen]
    last_end = max(n["onset"] + n["duration"] for n in gen)
    n_oor = sum(1 for p in pitches if p >= PITCH_HIGH or p <= PITCH_LOW)

    grid = rn_grid(all_notes, prompt_end, key_obj)
    dist = grid if keep_silent else [r for r in grid if r not in ("–", "?")]
    top1 = (max(Counter(dist).values()) / N_SLOTS) if dist else 0.0

    return {
        "struct_error": struct_err,               # None if no tokens.json (see docstring)
        "pitch_repeat": longest_run(pitches) >= REPEAT_RUN,
        "gen_wall_ms": last_end - prompt_end,
        "out_of_range": n_oor >= OOR_MIN,
        "durs": durs,
        "pitch_var": float(np.var(pitches)),
        "rn_entropy": shannon_bits(dist),
        "top1_dom": top1,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steering", default="reproduce_output/steering")
    p.add_argument("--steering_data", default="reproduce_output/steering_data")
    p.add_argument("--fmd", default="reproduce_output/fmd_cont_data/fmd_cont_scores.json")
    p.add_argument("--conditions", nargs="+", default=["baseline", "relative", "parallel"])
    p.add_argument("--fmd_suffix", default="",
                   help="appended to the FMD set name looked up in fmd_scores.json; "
                        "match build_fmd_data.py --set_suffix (e.g. '_l12')")
    p.add_argument("--keep_silent_slots", action="store_true",
                   help="count silent '–' slots as a category in the RN distribution")
    p.add_argument("--out", default="reproduce_output/steering/quality_metrics.csv")
    args = p.parse_args()

    fmd = json.load(open(args.fmd))["scores"] if os.path.exists(args.fmd) else {}

    tok = AbsTokenizer()
    key_cache, plen_cache = {}, {}
    rows = []
    for cond in args.conditions:
        outs = sorted(glob.glob(os.path.join(args.steering, cond, "*", "*", "*", "output.mid")))
        cells = []
        for out in outs:
            cut_stem = os.path.basename(os.path.dirname(out))
            chorale = os.path.basename(os.path.dirname(os.path.dirname(out)))
            split = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(out))))
            chords = os.path.join(args.steering_data, split, f"{chorale}_chords.txt")
            cut_mid = os.path.join(args.steering_data, split, f"{chorale}_cuts_output", f"{cut_stem}.mid")
            if chords not in key_cache:
                key_cache[chords] = parse_key(chords) if os.path.exists(chords) else None
            key_obj = key_cache[chords]
            if key_obj is None or not os.path.exists(cut_mid):
                continue
            r = analyze_cut(tok, out, cut_mid, key_obj, plen_cache, args.keep_silent_slots)
            if r:
                cells.append(r)

        n = len(cells)
        all_durs = [d for c in cells for d in c["durs"]]
        se = [c["struct_error"] for c in cells if c["struct_error"] is not None]
        row = {
            "condition": cond,
            "n": n,
            "struct_error_pct": (100 * np.mean(se)) if se else float("nan"),
            "pitch_repeat_pct": 100 * np.mean([c["pitch_repeat"] for c in cells]),
            "avg_gen_dur_ms": np.mean([c["gen_wall_ms"] for c in cells]),
            "out_of_range_pct": 100 * np.mean([c["out_of_range"] for c in cells]),
            "note_dur_avg_ms": np.mean(all_durs) if all_durs else float("nan"),
            "pitch_var": np.mean([c["pitch_var"] for c in cells]),
            "fmd": fmd.get(COND_TO_FMD[cond] + args.fmd_suffix, {}).get("fmd", float("nan")),
            "rn_entropy_bits": np.mean([c["rn_entropy"] for c in cells]),
            "top1_dom_pct": 100 * np.mean([c["top1_dom"] for c in cells]),
        }
        rows.append(row)

    cols = ["condition", "n", "struct_error_pct", "pitch_repeat_pct", "avg_gen_dur_ms",
            "out_of_range_pct", "note_dur_avg_ms", "pitch_var", "fmd",
            "rn_entropy_bits", "top1_dom_pct"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    # pretty print
    hdr = f"{'condition':10}{'n':>5}{'Struct%':>9}{'Rep%':>7}{'GenDur':>8}{'OOR%':>7}" \
          f"{'NoteDur':>9}{'PitchVar':>9}{'FMD':>9}{'RNent':>8}{'Top1%':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        se = "n/a" if math.isnan(r['struct_error_pct']) else f"{r['struct_error_pct']:.1f}"
        print(f"{r['condition']:10}{r['n']:>5}{se:>9}"
              f"{r['pitch_repeat_pct']:>7.1f}{r['avg_gen_dur_ms']:>8.0f}"
              f"{r['out_of_range_pct']:>7.1f}{r['note_dur_avg_ms']:>9.0f}"
              f"{r['pitch_var']:>9.1f}{r['fmd']:>9.2f}{r['rn_entropy_bits']:>8.2f}"
              f"{r['top1_dom_pct']:>8.1f}")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()

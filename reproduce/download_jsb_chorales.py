#!/usr/bin/env python3
"""
Download JSB Chorales dataset and convert to MIDI.

Source: https://github.com/czhuang/JSB-Chorales-dataset
The dataset is a piano-roll encoding of Bach chorales at 16th-note resolution.
This script converts it to MIDI files in the format expected by this repo.

Output: reproduce_output/jsb_chorales_midi/{train,valid,test}_16th/chorale_NNNN.mid

Usage:
    python reproduce/download_jsb_chorales.py
    python reproduce/download_jsb_chorales.py --output path/to/jsb_chorales_midi
"""

import argparse
import json
import os
import urllib.request

import mido

DATASET_URL = (
    "https://raw.githubusercontent.com/czhuang/JSB-Chorales-dataset"
    "/master/jsb-chorales-16th.json"
)

TICKS_PER_BEAT = 480
TICKS_PER_16TH = TICKS_PER_BEAT // 4   # 120
TEMPO = 500000                           # 120 BPM
VELOCITY = 64


def chorale_to_midi(chorale):
    """Convert a list of 16th-note chord tuples to a mido MidiFile."""
    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage("set_tempo", tempo=TEMPO, time=0))
    track.append(mido.Message("program_change", channel=0, program=0, time=0))

    events = []  # list of (abs_tick, type, note)

    prev_notes = set()
    for step, notes in enumerate(chorale):
        cur_notes = set(int(n) for n in notes)
        tick = step * TICKS_PER_16TH

        for note in sorted(prev_notes - cur_notes):
            events.append((tick, "note_off", note))
        for note in sorted(cur_notes - prev_notes):
            events.append((tick, "note_on", note))

        prev_notes = cur_notes

    # Close any remaining notes
    final_tick = len(chorale) * TICKS_PER_16TH
    for note in sorted(prev_notes):
        events.append((final_tick, "note_off", note))

    # Convert absolute ticks to delta times
    prev_tick = 0
    for abs_tick, msg_type, note in events:
        delta = abs_tick - prev_tick
        track.append(mido.Message(msg_type, channel=0, note=note,
                                  velocity=VELOCITY, time=delta))
        prev_tick = abs_tick

    track.append(mido.MetaMessage("end_of_track", time=0))
    return mid


def main():
    parser = argparse.ArgumentParser(description="Download and convert JSB Chorales to MIDI")
    parser.add_argument("--output", default="reproduce_output/jsb_chorales_midi",
                        help="Output directory (default: reproduce_output/jsb_chorales_midi)")
    args = parser.parse_args()

    print("Downloading JSB Chorales dataset...")
    with urllib.request.urlopen(DATASET_URL) as r:
        data = json.loads(r.read())

    for split, dirname in [("train", "train_16th"), ("valid", "valid_16th"), ("test", "test_16th")]:
        chorales = data[split]
        out_dir = os.path.join(args.output, dirname)
        os.makedirs(out_dir, exist_ok=True)
        print(f"Writing {len(chorales)} {split} chorales to {out_dir}/")
        for i, chorale in enumerate(chorales):
            mid = chorale_to_midi(chorale)
            path = os.path.join(out_dir, f"chorale_{i:04d}.mid")
            mid.save(path)
        print(f"  Done: {len(chorales)} files")

    total = sum(len(data[s]) for s in data)
    print(f"\nDone. {total} chorales written to {args.output}/")


if __name__ == "__main__":
    main()

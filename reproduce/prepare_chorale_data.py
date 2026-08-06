#!/usr/bin/env python3
"""
Prepare chorale data: harmonic analysis, cadence detection, MIDI cutting,
and minorization of cut points.

Three-step pipeline (run steps independently or all together):
  1. analyze   — generate _chords.txt with per-16th harmonic analysis
  2. cut       — cut MIDI at detected cadence points
  3. minorize  — lower major 3rd of final V chord in each cut

Usage:
    python reproduce/prepare_chorale_data.py analyze \
        --mode major --input reproduce_output/jsb_chorales_midi/train_16th \
        --output reproduce_output/data/major_chorale_corpus

    python reproduce/prepare_chorale_data.py cut \
        --input reproduce_output/jsb_chorales_midi/train_16th \
        --analysis reproduce_output/data/major_chorale_corpus \
        --output reproduce_output/data/major_chorale_corpus

    python reproduce/prepare_chorale_data.py minorize \
        --input reproduce_output/data/major_chorale_corpus

    python reproduce/prepare_chorale_data.py all \
        --mode major --input reproduce_output/jsb_chorales_midi/train_16th \
        --output reproduce_output/data/major_chorale_corpus
"""

import argparse
import os
import re
import sys

import mido
import music21.chord
import music21.converter
import music21.pitch
import music21.roman


# ── MIDI note extraction ──────────────────────────────────────────────────────

def _collect_midi_events(midi_path):
    """Return sorted list of (type, abs_tick, note) from all tracks."""
    mid = mido.MidiFile(midi_path)
    events = []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                events.append(('note_on', abs_tick, msg.note))
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                events.append(('note_off', abs_tick, msg.note))
    return sorted(events, key=lambda x: x[1])


def _get_sounding_notes_per_16th(midi_path):
    """
    Use mido to determine which MIDI notes sound at each 16th-note position.
    Returns (list of sorted MIDI note lists, ticks_per_16th).
    """
    mid = mido.MidiFile(midi_path)
    ticks_per_16th = mid.ticks_per_beat // 4

    note_starts = {}
    intervals = []
    abs_tick = 0
    for msg in mido.merge_tracks(mid.tracks):
        abs_tick += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            note_starts[msg.note] = abs_tick
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            if msg.note in note_starts:
                intervals.append((msg.note, note_starts.pop(msg.note), abs_tick))

    for note, start in note_starts.items():
        intervals.append((note, start, abs_tick))

    if not intervals:
        return [], ticks_per_16th

    n_steps = max(end for _, _, end in intervals) // ticks_per_16th
    result = [
        sorted(note for note, start, end in intervals if start <= step * ticks_per_16th < end)
        for step in range(n_steps)
    ]
    return result, ticks_per_16th




# ── Harmonic analysis ─────────────────────────────────────────────────────────

def _analyze_tick(notes):
    """Return (chord_name, roman_figure) for a set of MIDI notes."""
    pitches = [music21.pitch.Pitch(n) for n in notes]
    c = music21.chord.Chord(pitches)
    root = c.root().name.replace('-', 'b')
    chord_name = f"{root}-{c.commonName}"
    return chord_name, None  # Roman numeral filled in per-key in caller


def _write_analysis_file(output_path, midi_path, key_str, rows):
    """Write the tick-by-tick harmonic analysis to a text file."""
    lines = [
        f"File: {midi_path}",
        "Type: MIDI",
        f"Key: {key_str}",
        "-" * 60,
        f"{'Tick':<7}| {'Chord Name':<30} | Function",
        "-" * 60,
    ]
    for tick, chord_name, figure in rows:
        if chord_name is None:
            continue
        lines.append(f"{tick:<7}| {chord_name:<30} | {figure}")
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def save_harmonic_analysis(input_path, output_path, target_mode):
    """
    Analyze harmonic structure of a MIDI file and write _chords.txt.

    Args:
        input_path: path to .mid file
        output_path: path to write _chords.txt
        target_mode: 'major' or 'minor'
    """
    score = music21.converter.parse(input_path)
    key = score.analyze('key')
    key_str = str(key)

    if target_mode and key.mode != target_mode:
        return False

    notes_per_step, _ = _get_sounding_notes_per_16th(input_path)

    rows = []
    prev_chord_name, prev_figure = None, None
    for tick, notes in enumerate(notes_per_step):
        if len(notes) < 2:
            # Brief gap during note transition — carry forward previous chord
            rows.append((tick, prev_chord_name, prev_figure))
            continue
        pitches = [music21.pitch.Pitch(n) for n in notes]
        c = music21.chord.Chord(pitches)
        root = c.root().name.replace('-', 'b')
        chord_name = f"{root}-{c.commonName}"
        try:
            rn = music21.roman.romanNumeralFromChord(c, key)
            figure = rn.figure
        except Exception:
            figure = '?'
        prev_chord_name, prev_figure = chord_name, figure
        rows.append((tick, chord_name, figure))

    _write_analysis_file(output_path, input_path, key_str, rows)
    return True


# ── Cadence detection ─────────────────────────────────────────────────────────

def parse_harmonic_output(file_content):
    """Parse _chords.txt content into a list of {tick, function} dicts."""
    timeline = []
    for line in file_content.splitlines():
        m = re.match(r'^(\d+)\s*\|\s*(.+?)\s*\|\s*(\S+)', line)
        if m:
            timeline.append({
                'tick': int(m.group(1)),
                'chord': m.group(2).strip(),
                'function': m.group(3).strip(),
            })
    return timeline


def get_chord_events(timeline):
    """Compress per-tick timeline into chord events with durations."""
    if not timeline:
        return []
    events = []
    cur = dict(timeline[0])
    cur['duration'] = 1
    for entry in timeline[1:]:
        if entry['function'] == cur['function'] and entry['chord'] == cur['chord']:
            cur['duration'] += 1
        else:
            events.append(cur)
            cur = dict(entry)
            cur['duration'] = 1
    events.append(cur)
    return events


def _is_dominant(figure):
    """V, V7, V65, etc. — uppercase V only; not vi, vii, IV, or lowercase v."""
    if not figure:
        return False
    return figure.startswith('V') and not figure.startswith('VI') and not figure.startswith('VII')


def _is_tonic(figure, mode):
    """i or I but not ii, iii, iv, II, III, IV. Picardy I is allowed as a valid minor-key resolution."""
    if not figure:
        return False
    f = figure.upper()
    # Exclude ii, iii, iv and their inversions
    if f.startswith('II') or f.startswith('III') or f.startswith('IV'):
        return False
    if not f.startswith('I'):
        return False
    return True


def _is_root_position_tonic(figure):
    """True only for root-position tonic: I, i, i5, i52, i54 — not I6, i6, I64 etc."""
    return bool(re.match(r'^[Ii]5?[024]?$', figure))


def is_perfect_cadence(prev, curr, mode, ticks_per_beat=480, ticks_per_16th=120):
    """Check if prev → curr is a V → i/I cadence on a beat.

    The dominant must last at least two 16th-notes; a single-16th dominant is a
    passing/suspension figure (e.g. V743, V74#3) rather than a genuine cadential V.
    """
    if not (_is_dominant(prev['function']) and _is_tonic(curr['function'], mode)):
        return False
    if prev['duration'] < 2:
        return False
    ticks_per_beat_in_16ths = ticks_per_beat // ticks_per_16th  # 4
    return curr['tick'] % ticks_per_beat_in_16ths == 0


def _detect_mode(text_data):
    """Read mode from Key line in chords.txt."""
    for line in text_data.splitlines():
        if line.startswith('Key:'):
            return 'minor' if 'minor' in line else 'major'
    return 'major'


def find_cadences(text_data):
    """Return (strong_cadences, regular_cadences) tick lists."""
    mode = _detect_mode(text_data)
    timeline = parse_harmonic_output(text_data)
    events = get_chord_events(timeline)

    strong_cadences = []
    regular_cadences = []

    for i in range(1, len(events)):
        prev = events[i - 1]
        curr = events[i]
        if is_perfect_cadence(prev, curr, mode):
            tick = curr['tick']
            if curr['duration'] >= 8:
                strong_cadences.append(tick)
            else:
                regular_cadences.append(tick)

    return strong_cadences, regular_cadences


def save_cadences_to_file(chords_path):
    """Append cadence tick lists to an existing _chords.txt file."""
    with open(chords_path) as f:
        text = f.read()
    strong, regular = find_cadences(text)
    with open(chords_path, 'a') as f:
        f.write(f"Strong Cadences: {strong}\n")
        f.write(f"Regular Cadences: {regular}\n")


def extract_cut_points(analysis_text_path):
    """Parse strong cadence ticks from a _chords.txt file."""
    with open(analysis_text_path) as f:
        text = f.read()
    m = re.search(r'Strong Cadences.*?\[(.*?)\]', text)
    if not m or not m.group(1).strip():
        return []
    return [int(x.strip()) for x in m.group(1).split(',') if x.strip()]


# ── MIDI cutting ──────────────────────────────────────────────────────────────

def _update_active_notes(active, msg):
    if msg.type == 'note_on' and msg.velocity > 0:
        active.add(msg.note)
    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
        active.discard(msg.note)
    return active


def _handle_boundary_msg(msg, cut_tick):
    """At exact cut boundary: reject note-ons, keep note-offs."""
    if msg.type == 'note_on' and msg.velocity > 0:
        return None
    return msg


def _force_close_notes(active, abs_tick):
    """Force-close any notes still ringing at the cut point."""
    return [mido.Message('note_off', channel=0, note=n, velocity=64, time=0)
            for n in sorted(active)]


def _process_track_before_cut(track, cut_abs_tick):
    """Return (new_track, active_notes) with events up to cut boundary."""
    new_msgs = []
    active = set()
    abs_tick = 0
    prev_abs = 0

    for msg in track:
        abs_tick += msg.time
        if abs_tick > cut_abs_tick:
            break
        if abs_tick == cut_abs_tick:
            handled = _handle_boundary_msg(msg, cut_abs_tick)
            if handled is None:
                continue
            delta = abs_tick - prev_abs
            new_msgs.append(msg.copy(time=delta))
            prev_abs = abs_tick
            _update_active_notes(active, msg)
        else:
            delta = abs_tick - prev_abs
            new_msgs.append(msg.copy(time=delta))
            prev_abs = abs_tick
            _update_active_notes(active, msg)

    return new_msgs, active


def cut_midi_at_16th(midi_path, cut_16th_index, output_path):
    """
    Cut a MIDI file at a specific 16th-note boundary.

    NOTE: This force-closes notes still ringing at the cut, which ensures
    the output is a valid MIDI with no hanging notes.
    """
    mid = mido.MidiFile(midi_path)
    ticks_per_16th = mid.ticks_per_beat // 4
    cut_abs_tick = cut_16th_index * ticks_per_16th

    new_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
    new_track = mido.MidiTrack()
    new_mid.tracks.append(new_track)

    for track in mid.tracks:
        msgs, active = _process_track_before_cut(track, cut_abs_tick)
        for msg in msgs:
            if not isinstance(msg, mido.MetaMessage) or msg.type == 'set_tempo':
                new_track.append(msg)
        for close_msg in _force_close_notes(active, cut_abs_tick):
            new_track.append(close_msg)
        break  # single-track files

    new_track.append(mido.MetaMessage('end_of_track', time=0))
    new_mid.save(output_path)


# ── Minorization ──────────────────────────────────────────────────────────────

def _find_last_chord_notes(midi_path):
    """Return sorted MIDI note numbers of the last sounding chord."""
    mid = mido.MidiFile(midi_path)
    active = set()
    last_chord = set()
    for msg in mido.merge_tracks(mid.tracks):
        if msg.type == 'note_on' and msg.velocity > 0:
            active.add(msg.note)
            last_chord = set(active)
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            active.discard(msg.note)
    return sorted(last_chord)


def _find_major_third_notes(chord_notes):
    """Return MIDI note numbers that are the major 3rd of the chord."""
    if not chord_notes:
        return []
    root = chord_notes[0]
    major_third = root + 4
    return [n for n in chord_notes if n % 12 == major_third % 12]


def _find_note_onsets(midi_path, target_notes):
    """For each target note, find its last onset time before end."""
    mid = mido.MidiFile(midi_path)
    onsets = {}
    abs_tick = 0
    for msg in mido.merge_tracks(mid.tracks):
        abs_tick += msg.time
        if msg.type == 'note_on' and msg.velocity > 0 and msg.note in target_notes:
            onsets[msg.note] = abs_tick
    return onsets


def make_dominant_minor(input_path, output_path):
    """Lower the major 3rd of the final V chord by one semitone."""
    chord_notes = _find_last_chord_notes(input_path)
    if not chord_notes:
        return False

    major_thirds = _find_major_third_notes(chord_notes)
    if not major_thirds:
        return False

    target_set = set(major_thirds)
    onsets = _find_note_onsets(input_path, target_set)

    mid = mido.MidiFile(input_path)
    new_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
    new_track = mido.MidiTrack()
    new_mid.tracks.append(new_track)

    abs_tick = 0
    for msg in mido.merge_tracks(mid.tracks):
        abs_tick += msg.time
        if (msg.type in ('note_on', 'note_off') and
                msg.note in target_set and
                abs_tick >= onsets.get(msg.note, float('inf'))):
            new_track.append(msg.copy(note=msg.note - 1))
        else:
            new_track.append(msg)

    new_mid.save(output_path)
    return True


# ── Pipeline steps ────────────────────────────────────────────────────────────

def run_analyze(input_dir, output_dir, target_mode='major'):
    """Step 1: Generate _chords.txt for all MIDIs in input_dir."""
    os.makedirs(output_dir, exist_ok=True)
    processed, skipped = 0, 0
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith('.mid'):
            continue
        input_path = os.path.join(input_dir, fname)
        stem = os.path.splitext(fname)[0]
        output_path = os.path.join(output_dir, f"{stem}_chords.txt")
        ok = save_harmonic_analysis(input_path, output_path, target_mode)
        if ok:
            save_cadences_to_file(output_path)
            processed += 1
        else:
            skipped += 1
    print(f"  Analyze: {processed} chorales analyzed, {skipped} skipped (wrong mode)")


def _process_cadence_cuts(midi_path, analysis_dir, output_dir):
    """Cut one MIDI at its strong cadence points. Returns number of cuts made."""
    stem = os.path.splitext(os.path.basename(midi_path))[0]
    chords_path = os.path.join(analysis_dir, f"{stem}_chords.txt")
    if not os.path.exists(chords_path):
        return 0
    cut_points = extract_cut_points(chords_path)
    if not cut_points:
        return 0
    cuts_dir = os.path.join(output_dir, f"{stem}_cuts_output")
    os.makedirs(cuts_dir, exist_ok=True)
    for tick in cut_points:
        out_path = os.path.join(cuts_dir, f"{stem}_cut_tick_{tick}.mid")
        cut_midi_at_16th(midi_path, tick, out_path)
    return len(cut_points)


def run_cut(input_dir, analysis_dir, output_dir):
    """Step 2: Cut MIDIs at cadence points."""
    os.makedirs(output_dir, exist_ok=True)
    total_cuts = 0
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith('.mid'):
            continue
        total_cuts += _process_cadence_cuts(os.path.join(input_dir, fname), analysis_dir, output_dir)
    print(f"  Cut: {total_cuts} MIDI segments created")


def run_minorize(corpus_dir):
    """Step 3: Minorize all cut MIDIs in corpus subdirectories."""
    total, failed = 0, 0
    for entry in sorted(os.listdir(corpus_dir)):
        if not entry.endswith('_cuts_output'):
            continue
        cuts_dir = os.path.join(corpus_dir, entry)
        minor_dir = os.path.join(cuts_dir, 'minorized')
        os.makedirs(minor_dir, exist_ok=True)
        for fname in sorted(os.listdir(cuts_dir)):
            if not fname.endswith('.mid'):
                continue
            in_path = os.path.join(cuts_dir, fname)
            stem = os.path.splitext(fname)[0]
            out_path = os.path.join(minor_dir, f"{stem}_minor.mid")
            ok = make_dominant_minor(in_path, out_path)
            if ok:
                total += 1
            else:
                failed += 1
    print(f"  Minorize: {total} segments minorized" + (f", {failed} failed (no major 3rd)" if failed else ""))


def _default_output(mode):
    return 'major_corpus' if mode == 'major' else 'minor_corpus'


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser():
    """Prepare chorale data: analyze, cut, minorize."""
    p = argparse.ArgumentParser(description="Prepare chorale data: analyze, cut, minorize.")
    sub = p.add_subparsers(dest='command')

    # analyze
    a = sub.add_parser('analyze', help='Generate _chords.txt files')
    a.add_argument('--mode', default='major', help='major or minor')
    a.add_argument('--input', dest='midi_dir', required=True, help='Dir with .mid files')
    a.add_argument('--output', dest='output_dir', required=True, help='Dir for _chords.txt output')

    # cut
    c = sub.add_parser('cut', help='Cut MIDIs at cadence points')
    c.add_argument('--input', dest='midi_dir', required=True)
    c.add_argument('--analysis', dest='analysis_dir', required=True, help='Dir with _chords.txt files')
    c.add_argument('--output', dest='output_dir', required=True, help='Dir for cut output')

    # minorize
    m = sub.add_parser('minorize', help='Lower major 3rd of final V chord')
    m.add_argument('--input', dest='corpus_dir', required=True, help='Corpus dir with _cuts_output/ subdirs')

    # all
    al = sub.add_parser('all', help='Run full pipeline: analyze + cut + minorize')
    al.add_argument('--mode', default='major')
    al.add_argument('--input', dest='midi_dir', required=True)
    al.add_argument('--output', dest='output_dir', required=True, help='Dir for all output')

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == 'analyze':
        run_analyze(args.midi_dir, args.output_dir, args.mode)

    elif args.command == 'cut':
        run_cut(args.midi_dir, args.analysis_dir, args.output_dir)

    elif args.command == 'minorize':
        run_minorize(args.corpus_dir)

    elif args.command == 'all':
        print(f"Preparing {args.mode} chorale corpus → {args.output_dir}")
        run_analyze(args.midi_dir, args.output_dir, args.mode)
        run_cut(args.midi_dir, args.output_dir, args.output_dir)
        run_minorize(args.output_dir)
        print("Done.")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()

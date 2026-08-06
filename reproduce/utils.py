"""Shared utilities for experiment analysis scripts."""

import json
import os

from music21 import pitch, chord, roman, key as m21key

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def midi_to_note_name(midi_num: int) -> str:
    """Convert MIDI note number to note name (e.g., 60 -> C4, 69 -> A4)."""
    octave = (midi_num // 12) - 1
    note = NOTE_NAMES[midi_num % 12]
    return f"{note}{octave}"


def decode_tokens_to_notes(token_ids, positions, tokenizer):
    """
    Decode token IDs into note dicts with keys:
        midi, velocity, onset (ms), duration (ms), pitch_position, pitch_token_id
    Tracks <T> tokens to compute absolute onset.
    """
    tokens = [(tokenizer.id_to_tok[tid], pos, tid) for tid, pos in zip(token_ids, positions)]
    time_tok_id = tokenizer.tok_to_id[tokenizer.time_tok]
    time_offset = 0
    notes = []
    i = 0
    while i < len(tokens):
        tok, pos, tid = tokens[i]
        if tid == time_tok_id:
            time_offset += tokenizer.abs_time_step_ms
            i += 1
            continue
        if isinstance(tok, tuple) and tok[0] == 'piano':
            onset = dur = None
            if i + 1 < len(tokens) and isinstance(tokens[i+1][0], tuple) and tokens[i+1][0][0] == 'onset':
                onset = tokens[i+1][0][1]
            if i + 2 < len(tokens) and isinstance(tokens[i+2][0], tuple) and tokens[i+2][0][0] == 'dur':
                dur = tokens[i+2][0][1]
            if onset is not None and dur is not None:
                notes.append({
                    'midi': tok[1], 'velocity': tok[2],
                    'onset': time_offset + onset, 'duration': dur,
                    'pitch_position': pos, 'pitch_token_id': tid,
                })
                i += 3
                continue
        i += 1
    return notes


def _identify_chord(midi_notes, parsed_key):
    """
    Given a list of MIDI note numbers and a music21 Key, return
    (chord_name, roman_numeral) or (None, None) if no notes.
    """
    if not midi_notes or len(midi_notes) < 2:
        return None, None

    # Default spelling (music21 picks flats for some notes, e.g. Bb instead of A#).
    c = chord.Chord([pitch.Pitch(midi=m) for m in midi_notes])
    chord_name = c.pitchedCommonName

    # If music21 can't name the chord directly (enharmonic fallback), retry with
    # sharp spelling.  e.g. F#-Bb-Db → F#-A#-C# = F# major triad.
    if 'enharmonic' in chord_name.lower():
        c = chord.Chord([
            pitch.Pitch(f"{NOTE_NAMES[m % 12]}{(m // 12) - 1}")
            for m in midi_notes
        ])
        chord_name = c.pitchedCommonName

    rn = roman.romanNumeralFromChord(c, parsed_key)
    figure = rn.figure

    # music21 uses the ascending melodic minor scale (raised 6th and 7th) as
    # its reference for roman numeral labeling in minor keys.  This means
    # chords built on the natural minor's 6th and 7th degrees receive a
    # spurious 'b' prefix (e.g. bVI instead of VI for a diatonic Eb major
    # triad in G minor).  We strip it so that roman numerals reflect the
    # natural minor scale, which is the standard reference for Bach chorales.
    if parsed_key.mode == 'minor':
        for prefix, replacement in [('bVII', 'VII'), ('bvii', 'vii'),
                                    ('bVI', 'VI'), ('bvi', 'vi')]:
            if figure.startswith(prefix):
                figure = replacement + figure[len(prefix):]
                break

    return chord_name, figure


def get_patch_note(seed_dir):
    """Return info about which note was patched before generation.

    Given a seed directory (e.g. .../chorale_0176/cut_tick_24/seed_42)
    that contains original_activations/ and patched_activations/,
    finds the prompt token position where the KV-cache was patched from the
    minorized run, and returns a dict with:
        patch_position      – token index where patching was done
        original_midi       – MIDI note number in the original sequence
        original_note_name  – note name in the original (e.g. 'A3')
        patched_midi        – MIDI note number from minorized (the replacement)
        patched_note_name   – note name from minorized (e.g. 'Ab3')
    Returns None if data cannot be found or no divergence is detected.
    """
    from ariautils.tokenizer import AbsTokenizer

    # Find experiment_info.json by walking up (up to 5 levels)
    search = seed_dir
    info = None
    for _ in range(5):
        search = os.path.dirname(search)
        candidate = os.path.join(search, 'experiment_info.json')
        if os.path.exists(candidate):
            with open(candidate) as f:
                info = json.load(f)
            break
    if info is None:
        return None

    capture_generated_tokens = info['capture_generated_tokens']

    orig_meta_path = os.path.join(seed_dir, 'original_activations', 'metadata.json')
    patch_meta_path = os.path.join(seed_dir, 'patched_activations', 'metadata.json')
    if not os.path.exists(orig_meta_path) or not os.path.exists(patch_meta_path):
        return None

    with open(orig_meta_path) as f:
        orig_meta = json.load(f)
    with open(patch_meta_path) as f:
        patch_meta = json.load(f)

    tokenizer = AbsTokenizer()
    orig_positions = orig_meta['capture_positions']
    orig_token_ids = orig_meta['token_ids']
    patch_positions = patch_meta['capture_positions']
    patch_token_ids = patch_meta['token_ids']

    n_gen = min(capture_generated_tokens, len(orig_positions))
    prompt_end_position = orig_positions[-n_gen]

    orig_map = {pos: tid for pos, tid in zip(orig_positions, orig_token_ids)}
    patch_map = {pos: tid for pos, tid in zip(patch_positions, patch_token_ids)}

    # Find the prompt position where original and patched token IDs differ.
    # That position is the patching (divergence) point.
    for pos in sorted(set(orig_map) & set(patch_map)):
        if pos >= prompt_end_position:
            continue
        orig_tid = orig_map[pos]
        patch_tid = patch_map[pos]
        if orig_tid == patch_tid:
            continue
        orig_tok = tokenizer.id_to_tok[orig_tid]
        patch_tok = tokenizer.id_to_tok[patch_tid]
        if not (isinstance(orig_tok, tuple) and orig_tok[0] == 'piano'):
            continue
        if not (isinstance(patch_tok, tuple) and patch_tok[0] == 'piano'):
            continue
        # Count notes in the prompt strictly after this patch position
        _after_ids = [tid for p, tid in zip(orig_positions, orig_token_ids)
                      if p > pos and p < prompt_end_position]
        _after_pos = [p for p in orig_positions
                      if p > pos and p < prompt_end_position]
        notes_after = decode_tokens_to_notes(_after_ids, _after_pos, tokenizer)

        return {
            'patch_position': pos,
            'original_midi': orig_tok[1],
            'original_note_name': midi_to_note_name(orig_tok[1]),
            'patched_midi': patch_tok[1],
            'patched_note_name': midi_to_note_name(patch_tok[1]),
            'notes_after_patch': notes_after,
        }

    return None


def tok_to_str(tok):
    """Human-readable string for a tokenizer token (tuple or string)."""
    if isinstance(tok, tuple) and tok[0] == 'piano':
        return f"piano {midi_to_note_name(tok[1])} vel={tok[2]}"
    if isinstance(tok, tuple):
        return f"{tok[0]} {tok[1]}"
    return str(tok)


def build_token_translation(token_ids, tokenizer):
    """Return list of human-readable strings for a list of token IDs."""
    return [tok_to_str(tokenizer.id_to_tok[tid]) for tid in token_ids]


def _parse_key_from_chords_txt(chords_path):
    """Parse 'Key: B- major' line from a _chords.txt file into a music21 Key."""
    with open(chords_path, 'r') as f:
        for line in f:
            if line.startswith('Key:'):
                # e.g. "Key: B- major"
                parts = line[4:].strip().split()
                tonic_name = parts[0]  # e.g. "B-"
                mode = parts[1] if len(parts) > 1 else "major"
                return m21key.Key(tonic_name, mode)
    return None


def _find_continuation_notes(prompt_notes, generated_notes):
    """Split generated notes into (overlap, continuation).

    continuation — the notes at the first onset strictly after the last prompt
    onset; if there are no prompt notes (only the patch position was captured),
    fall back to the first onset of the generated notes. This is the single
    "resulting harmony" chord. Ported verbatim from experiments/analyze_harmony.py.
    """
    last_onset = max((n['onset'] for n in prompt_notes), default=None)
    if last_onset is None:
        first_onset = min((n['onset'] for n in generated_notes), default=None)
        return [], [n for n in generated_notes if n['onset'] == first_onset] if first_onset is not None else []

    gen_after = [n for n in generated_notes if n['onset'] > last_onset]
    first_cont_onset = min((n['onset'] for n in gen_after), default=None)
    return [], [n for n in gen_after if n['onset'] == first_cont_onset] if first_cont_onset is not None else []


def resulting_harmony_notes(metadata, tokenizer, n_gen=None):
    """Notes of the first continuation chord for one variant, ordered by position.

    Mirrors experiments/analyze_harmony.analyze_variant: decode the captured
    tokens, split prompt/generated at capture_positions[-n_gen], and keep the
    notes of the first onset after the prompt. Equivalent to that function's
    'all_generated_notes' (and 'notes', up to ordering). Empty list if none.

    Every capture list is the patch position followed by the generated tokens, so
    n_gen defaults to len(capture_positions) - 1. Pass it explicitly only to
    override; a value longer than the capture list is clamped, which matters for
    the per-layer patch dirs (they capture fewer generated tokens than the
    original/minorized baselines).
    """
    capture_positions = metadata['capture_positions']
    n_gen = min(n_gen or len(capture_positions) - 1, len(capture_positions) - 1)
    prompt_len = capture_positions[-n_gen]
    all_notes = decode_tokens_to_notes(metadata['token_ids'], capture_positions, tokenizer)
    prompt_notes = [n for n in all_notes if n['pitch_position'] < prompt_len]
    generated_notes = [n for n in all_notes if n['pitch_position'] >= prompt_len]
    _, continuation_notes = _find_continuation_notes(prompt_notes, generated_notes)
    return sorted(continuation_notes, key=lambda n: n['pitch_position'])


def resulting_harmony_rn(metadata, parsed_key, tokenizer, n_gen=None):
    """Roman numeral of the first continuation chord for one variant's metadata.

    Labels the chord returned by `resulting_harmony_notes` relative to
    `parsed_key`. Returns the roman-numeral figure (e.g. 'I64', 'vi'), or None
    if the continuation has no notes.
    """
    continuation_notes = resulting_harmony_notes(metadata, tokenizer, n_gen)
    if not continuation_notes:
        return None
    midi_notes = sorted(set(n['midi'] for n in continuation_notes))
    _, rn = _identify_chord(midi_notes, parsed_key)
    return rn

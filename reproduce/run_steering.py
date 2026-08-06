#!/usr/bin/env python3
"""
Run a steering experiment on the V->I major-chorale cuts (paper Section 5.3, Fig 5).

Adds probe-derived steering vectors (Step 11, reproduce_output/steering/directions.pt)
to the residual stream while generating a continuation for each Step-10 cut MIDI.
Modular over the steering condition:

    mode      key-independent direction (toward minor); last-bar prompt positions
    relative  per-key direction toward the relative minor; all positions incl. generated
    parallel  per-key direction toward the parallel minor; all positions incl. generated
    baseline  no steering (unsteered continuation, for the Figure-5 reference bar)

Each condition has default (layers, alpha, position) matching the chosen settings; any
can be overridden on the command line. This step only GENERATES the continuations
(one output.mid per cut) — it does not label resolutions.

REQUIRES A GPU.

Usage:
    python reproduce/run_steering.py --condition mode
    python reproduce/run_steering.py --condition relative --alpha 0.2
    python reproduce/run_steering.py --condition baseline
"""

import argparse
import glob
import json
import os
import sys

import mido
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from consts import (
    CONTINUATION_SECONDS, DTYPE, MAX_GEN_LENGTH, MIN_GEN_LENGTH, MIN_P, TEMP,
)
from utils import decode_tokens_to_notes
from run_per_layer_patching import (
    DEVICE, load_model_and_tokenizer, _setup_cache, _set_seed, _sdpa_decode,
)

BAR_BEATS = 4

# Per-condition defaults: (steer layers, alpha, position condition).
CONDITION_DEFAULTS = {
    "mode":     {"layers": [5, 6, 7, 8, 9, 10],      "alpha": 0.25, "position": "last_bar"},
    "relative": {"layers": [11, 12, 13, 14, 15],     "alpha": 0.15, "position": "all_positions_and_gen"},
    "parallel": {"layers": [11, 12, 13, 14, 15],     "alpha": 0.10, "position": "all_positions_and_gen"},
    "baseline": {"layers": [],                        "alpha": 0.0,  "position": "last_bar"},
}


# ── steering controller ──────────────────────────────────────────────────────────

class SteeringController:
    """Adds per-layer vectors to the residual at selected absolute positions."""

    def __init__(self, model, layer_vectors, steer_positions):
        self.model = model
        self.layer_vectors = layer_vectors        # {layer: tensor[d_model]} (alpha-scaled)
        self.steer_positions = steer_positions    # set of absolute token positions
        self.current_positions = None             # positions in the current forward call
        self.handles = []

    def set_current_positions(self, positions):
        self.current_positions = positions

    def _make_hook(self, L):
        def hook(module, inp, out):
            if self.current_positions is None:
                return out
            v = self.layer_vectors[L].to(out.device, dtype=out.dtype)
            for i, pos in enumerate(self.current_positions):
                if pos in self.steer_positions:
                    out[0, i, :] = out[0, i, :] + v
            return out
        return hook

    def attach(self):
        for L in self.layer_vectors:
            self.handles.append(
                self.model.model.encode_layers[L].register_forward_hook(self._make_hook(L)))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ── positions ────────────────────────────────────────────────────────────────────

def last_bar_notes(notes, bar_duration_ms):
    """Prompt notes whose onset falls in the last bar."""
    if not notes:
        return []
    t_end = max(n["onset"] for n in notes)
    return [n for n in notes if n["onset"] >= t_end - bar_duration_ms]


def positions_for_last_bar(notes_in_bar):
    """All three token positions (pitch/onset/dur) of each note in the last bar."""
    s = set()
    for n in notes_in_bar:
        p = n["pitch_position"]
        s.update({p, p + 1, p + 2})
    return s


def compute_steer_positions(cond, prompt_len, total_len, notes_in_bar):
    gen = set(range(prompt_len, total_len))
    if cond == "last_position":
        return {prompt_len - 1}
    if cond == "last_position_and_gen":
        return {prompt_len - 1} | gen
    if cond == "last_bar":
        return positions_for_last_bar(notes_in_bar)
    if cond == "last_bar_and_gen":
        return positions_for_last_bar(notes_in_bar) | gen
    if cond == "all_positions":
        return set(range(prompt_len))
    if cond == "all_positions_and_gen":
        return set(range(total_len))
    raise ValueError(f"unknown position condition {cond!r}")


# ── prompt / generation ──────────────────────────────────────────────────────────

def bar_duration_ms(midi_path):
    """Milliseconds in one bar (BAR_BEATS beats) from the MIDI tempo."""
    m = mido.MidiFile(midi_path)
    tempo = 500000
    for track in m.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                tempo = msg.tempo
                break
    return BAR_BEATS * (tempo / 1000.0)


def tokenize_prompt(midi_path, tokenizer):
    from ariautils.midi import MidiDict
    md = MidiDict.from_midi(midi_path)
    tokens = tokenizer.tokenize(md, add_dim_tok=False, add_eos_tok=False)
    return tokenizer.encode(tokens)


def compute_gen_length(prompt_ids, notes):
    """Token budget for a CONTINUATION_SECONDS continuation, from prompt token density."""
    if not notes:
        return MIN_GEN_LENGTH
    dur = max(n["onset"] + n["duration"] for n in notes) / 1000.0
    if dur <= 0:
        return MIN_GEN_LENGTH
    per_sec = len(prompt_ids) / dur
    return int(min(MAX_GEN_LENGTH, max(MIN_GEN_LENGTH, per_sec * CONTINUATION_SECONDS)))


@torch.autocast(DEVICE, dtype=DTYPE)
@torch.inference_mode()
def generate_steered(model, tokenizer, prompt_ids, gen_len, controller):
    """Generate a continuation, optionally with the steering controller attached."""
    from aria.inference import sample_min_p
    from aria.inference.sample_cuda import prefill, update_seq_ids_

    pad_id = tokenizer.encode([tokenizer.pad_tok])[0]
    prompt_len = len(prompt_ids)
    total_len = prompt_len + gen_len
    _setup_cache(model, batch_size=1, max_seq_len=total_len, dtype=DTYPE)

    seq = torch.full((1, total_len), pad_id, dtype=torch.long, device=DEVICE)
    seq[0, :prompt_len] = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE)

    if controller is not None:
        controller.attach()
    try:
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            if controller is not None:
                controller.set_current_positions(list(range(prompt_len)))
            logits = prefill(model, idxs=seq[:, :prompt_len],
                             input_pos=torch.arange(prompt_len, dtype=torch.int, device=DEVICE))
            first_logits = logits[:, -1] if logits.dim() == 3 else logits

            dim_done, eos_done = [False], [False]
            for idx in range(prompt_len, total_len):
                if idx == prompt_len:
                    logits = first_logits
                else:
                    if controller is not None:
                        controller.set_current_positions([idx - 1])
                    logits = _sdpa_decode(model, seq, idx - 1)
                probs = torch.softmax(logits / TEMP, dim=-1)
                next_ids = sample_min_p(probs, MIN_P).flatten()
                update_seq_ids_(seq=seq, idx=idx, next_token_ids=next_ids,
                                dim_tok_inserted=dim_done, eos_tok_seen=eos_done,
                                max_len=total_len, force_end=False, tokenizer=tokenizer)
                if all(eos_done):
                    break
    finally:
        if controller is not None:
            controller.detach()
    return seq


def save_output(tokenizer, seq, prompt_len, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ids = seq[0].tolist()
    eos_id = tokenizer.encode([tokenizer.eos_tok])[0]
    if eos_id in ids:
        ids = ids[:ids.index(eos_id) + 1]
    # Raw generated token ids (including any malformed structure). Needed for the
    # structural-error metric, which a MIDI round-trip would silently discard.
    with open(os.path.join(out_dir, "tokens.json"), "w") as f:
        json.dump({"token_ids": ids, "prompt_len": prompt_len}, f)
    decoded = tokenizer.decode(ids)
    if tokenizer.eos_tok in decoded:
        decoded = decoded[:decoded.index(tokenizer.eos_tok) + 1]
    tokenizer.detokenize(decoded).to_midi().save(os.path.join(out_dir, "output.mid"))


# ── direction lookup ─────────────────────────────────────────────────────────────

def read_key(chords_path):
    for ln in open(chords_path):
        if ln.startswith("Key:"):
            return ln[len("Key:"):].strip()
    return None


def layer_vectors_for(directions, cond, major_key, alpha, layers):
    """Return {layer: alpha * g_layer}, or None if this key has no direction."""
    if cond == "baseline":
        return {}
    if cond == "mode":
        src = directions["mode"]
    else:
        if major_key not in directions[cond]:
            return None
        src = directions[cond][major_key]
    return {L: alpha * src[L] for L in layers}


# ── main ─────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True,
                   choices=["mode", "relative", "parallel", "baseline"])
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--layers", type=int, nargs="+", default=None)
    p.add_argument("--position", default=None,
                   choices=["last_position", "last_position_and_gen", "last_bar",
                            "last_bar_and_gen", "all_positions", "all_positions_and_gen"])
    p.add_argument("--splits", nargs="+", default=["test", "eval"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--directions", default="reproduce_output/steering/directions.pt")
    p.add_argument("--steering_data", default="reproduce_output/steering_data")
    p.add_argument("--output", default="reproduce_output/steering")
    p.add_argument("--limit", type=int, default=None, help="max cuts per split (smoke test)")
    args = p.parse_args()

    d = CONDITION_DEFAULTS[args.condition]
    alpha = d["alpha"] if args.alpha is None else args.alpha
    layers = d["layers"] if args.layers is None else args.layers
    position = d["position"] if args.position is None else args.position

    directions = torch.load(args.directions, map_location=DEVICE, weights_only=False) \
        if args.condition != "baseline" else {}
    model, tokenizer = load_model_and_tokenizer()
    _set_seed(args.seed)

    out_root = os.path.join(args.output, args.condition)
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "run_config.json"), "w") as f:
        json.dump({"condition": args.condition, "alpha": alpha, "layers": layers,
                   "position": position, "seed": args.seed, "splits": args.splits,
                   "temp": TEMP, "min_p": MIN_P}, f, indent=2)

    print(f"condition={args.condition}  alpha={alpha}  layers={layers}  position={position}")
    for split in args.splits:
        cuts = sorted(glob.glob(os.path.join(
            args.steering_data, split, "*_cuts_output", "*.mid")))
        if args.limit:
            cuts = cuts[:args.limit]
        print(f"\n=== {split}: {len(cuts)} cuts ===")
        done = skipped = 0
        for midi in cuts:
            stem = os.path.splitext(os.path.basename(midi))[0]
            chorale = stem.split("_cut_tick_")[0]
            out_dir = os.path.join(out_root, split, chorale, stem)
            if (os.path.exists(os.path.join(out_dir, "output.mid"))
                    and os.path.exists(os.path.join(out_dir, "tokens.json"))):
                continue

            major_key = read_key(os.path.join(args.steering_data, split,
                                              f"{chorale}_chords.txt"))
            vecs = layer_vectors_for(directions, args.condition, major_key, alpha, layers)
            if vecs is None:
                skipped += 1
                continue

            prompt_ids = tokenize_prompt(midi, tokenizer)
            notes = decode_tokens_to_notes(prompt_ids, list(range(len(prompt_ids))), tokenizer)
            gen_len = compute_gen_length(prompt_ids, notes)
            steer_pos = compute_steer_positions(
                position, len(prompt_ids), len(prompt_ids) + gen_len,
                last_bar_notes(notes, bar_duration_ms(midi)))
            controller = SteeringController(model, vecs, steer_pos) if vecs else None

            seq = generate_steered(model, tokenizer, prompt_ids, gen_len, controller)
            save_output(tokenizer, seq, len(prompt_ids), out_dir)
            done += 1
        print(f"    {done} generated, {skipped} skipped (no direction for key)")


if __name__ == "__main__":
    main()

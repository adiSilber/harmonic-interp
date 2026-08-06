#!/usr/bin/env bash
#
# Step 8 — Extract residual-stream activations for the probing experiments.
#
# For every bar-truncated MIDI (chorale_XXXX_bars_1-N.mid) written by Step 7,
# runs one forward pass through Aria and saves the residual stream at the LAST
# token of the sequence, for all 16 layers.
#
# Output (one file per chorale dir):
#   reproduce_output/probes_data/<split>/chorale_XXXX/activations.pt
#     {'activations': float32 [n_bars, n_layers, d_model], 'bar_indices': [...]}
#
# REQUIRES A GPU.
#
# Usage:
#     reproduce/extract_probe_activations.sh              # both splits; auto-splits
#                                                         #   train->GPU0, test->GPU1
#                                                         #   in parallel if 2+ GPUs
#     reproduce/extract_probe_activations.sh train        # only train (one GPU)
#     reproduce/extract_probe_activations.sh test         # only test  (one GPU)
#     reproduce/extract_probe_activations.sh --overwrite  # recompute existing activations.pt
#
# When run in parallel, each split logs to reproduce_output/probes_data/step8_<split>.log
# (tail -f them to watch). Set SERIAL=1 to force one-GPU sequential execution.
# Activate the aria conda env first, or override with:  PYTHON=/path/to/python reproduce/...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"

# Split args into split names (train/test) and passthrough flags (e.g. --overwrite).
SPLITS=()
EXTRA=()
for arg in "$@"; do
    case "$arg" in
        train|test) SPLITS+=("$arg") ;;
        *)          EXTRA+=("$arg") ;;
    esac
done
if [ "${#SPLITS[@]}" -eq 0 ]; then
    SPLITS=(train test)
fi

# Run one split. Args: <split> [gpu-index]. If gpu-index is given, pin to it;
# otherwise inherit the ambient CUDA_VISIBLE_DEVICES.
run_one() {
    local split="$1"
    local gpu="${2:-}"
    echo "[${split}] GPU ${gpu:-<inherited>}"
    if [ -n "$gpu" ]; then
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" reproduce/extract_probe_activations.py \
            --splits "$split" "${EXTRA[@]+"${EXTRA[@]}"}"
    else
        "$PYTHON" reproduce/extract_probe_activations.py \
            --splits "$split" "${EXTRA[@]+"${EXTRA[@]}"}"
    fi
}

N_GPU="$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)"

if [ "${#SPLITS[@]}" -ge 2 ] && [ "${N_GPU}" -ge 2 ] && [ "${SERIAL:-0}" != "1" ]; then
    echo "Two splits on ${N_GPU} GPUs — running in parallel (train->GPU0, test->GPU1)."
    echo "Logs: reproduce_output/probes_data/step8_{train,test}.log  (tail -f to watch)"
    run_one train 0 > reproduce_output/probes_data/step8_train.log 2>&1 &
    pid_train=$!
    run_one test 1 > reproduce_output/probes_data/step8_test.log 2>&1 &
    pid_test=$!
    wait "$pid_train" "$pid_test"
else
    # Sequential: each split on GPU 0 (or whatever CUDA_VISIBLE_DEVICES already selects).
    for split in "${SPLITS[@]}"; do
        echo "=================================================================="
        echo "Extracting probe activations: ${split}"
        echo "=================================================================="
        run_one "$split"
    done
fi

echo "Done. Activations written to reproduce_output/probes_data/<split>/chorale_*/activations.pt"

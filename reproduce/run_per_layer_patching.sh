#!/usr/bin/env bash
#
# Step 5 — Per-layer residual-stream patching (phase 2).
#
# For each (chorale, cut, seed) in a Step 3 activation run, and for each layer i,
# patches the residual stream at the last prompt position with the saved MINORIZED
# hidden state at layer i, then generates a continuation. The prompt KV is
# recomputed once per seed via a prefill (no kv_cache.pt from Step 3 needed).
#
# Output (new directories only), under each seed dir:
#   per_layer_last_position_patching/layer_<i>_patch/{output.mid, activations/}
#
# REQUIRES A GPU.
#
# Usage:
#     reproduce/run_per_layer_patching.sh                 # both corpora; auto-splits
#                                                         #   major->GPU0, minor->GPU1
#                                                         #   in parallel if 2+ GPUs
#     reproduce/run_per_layer_patching.sh major           # only major (one GPU)
#     reproduce/run_per_layer_patching.sh minor           # only minor (one GPU)
#
# When run in parallel, each corpus logs to reproduce_output/patching/step5_<corpus>.log
# (tail -f them to watch). Set SERIAL=1 to force one-GPU sequential execution.
# Activate the aria conda env first, or override with:  PYTHON=/path/to/python reproduce/...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"

# Split args into corpus modes (major/minor) and passthrough flags.
MODES=()
EXTRA=()
for arg in "$@"; do
    case "$arg" in
        major|minor) MODES+=("$arg") ;;
        *)           EXTRA+=("$arg") ;;
    esac
done
if [ "${#MODES[@]}" -eq 0 ]; then
    MODES=(major minor)
fi

# Run one corpus. Args: <mode> [gpu-index]. If gpu-index is given, pin to it;
# otherwise inherit the ambient CUDA_VISIBLE_DEVICES.
run_one() {
    local mode="$1"
    local gpu="${2:-}"
    local run_dir
    run_dir="$(ls -d reproduce_output/patching/*_reproduce_${mode} 2>/dev/null | sort | tail -1)"
    if [ -z "$run_dir" ]; then
        echo "No Step 3 run found for ${mode} (reproduce_output/patching/*_reproduce_${mode}); run Step 3 first." >&2
        return 0
    fi
    echo "[${mode}] GPU ${gpu:-<inherited>}  ->  ${run_dir}"
    if [ -n "$gpu" ]; then
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" reproduce/run_per_layer_patching.py \
            --experiment_dir "$run_dir" "${EXTRA[@]+"${EXTRA[@]}"}"
    else
        "$PYTHON" reproduce/run_per_layer_patching.py \
            --experiment_dir "$run_dir" "${EXTRA[@]+"${EXTRA[@]}"}"
    fi
}

N_GPU="$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)"

if [ "${#MODES[@]}" -ge 2 ] && [ "${N_GPU}" -ge 2 ] && [ "${SERIAL:-0}" != "1" ]; then
    echo "Two corpora on ${N_GPU} GPUs — running in parallel (major->GPU0, minor->GPU1)."
    echo "Logs: reproduce_output/patching/step5_{major,minor}.log  (tail -f to watch)"
    run_one major 0 > reproduce_output/patching/step5_major.log 2>&1 &
    pid_major=$!
    run_one minor 1 > reproduce_output/patching/step5_minor.log 2>&1 &
    pid_minor=$!
    wait "$pid_major" "$pid_minor"
else
    # Sequential: each corpus on GPU 0 (or whatever CUDA_VISIBLE_DEVICES already selects).
    for mode in "${MODES[@]}"; do
        echo "=================================================================="
        echo "Per-layer patching: ${mode}"
        echo "=================================================================="
        run_one "$mode"
    done
fi

echo "Done. Patching outputs written under each seed_*/per_layer_last_position_patching/"

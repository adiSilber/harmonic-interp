#!/usr/bin/env bash
#
# Step 12 — Steered generation for the Figure-5 experiments.
#
# Adds the Step-11 steering directions to Aria's residual stream while generating
# a continuation for each Step-10 V->I cut. One output.mid per cut. Runs all four
# conditions by default (baseline + the three steered ones), or a chosen subset.
#
# Output: reproduce_output/steering/<condition>/<split>/<chorale>/<cut>/output.mid
#
# REQUIRES A GPU.
#
# Usage:
#     reproduce/run_steering.sh                    # all: baseline mode relative parallel
#     reproduce/run_steering.sh mode               # a single condition
#     reproduce/run_steering.sh baseline mode      # a subset
#     reproduce/run_steering.sh mode --alpha 0.3   # extra flags pass through
#
# Each condition uses both GPUs when available (test -> GPU 0, eval -> GPU 1 in
# parallel), and the conditions run one after another. Each split logs to
# reproduce_output/steering/<condition>/step12_<split>.log. Set SERIAL=1 to force
# one-GPU sequential. Activate the aria conda env first, or set PYTHON=/path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"

# Split args into conditions and passthrough flags.
CONDITIONS=()
EXTRA=()
for arg in "$@"; do
    case "$arg" in
        baseline|mode|relative|parallel) CONDITIONS+=("$arg") ;;
        all)                             CONDITIONS=(baseline mode relative parallel) ;;
        *)                               EXTRA+=("$arg") ;;
    esac
done
if [ "${#CONDITIONS[@]}" -eq 0 ]; then
    CONDITIONS=(baseline mode relative parallel)
fi

# Honor a passthrough --output so a debug run keeps its outputs AND logs out of the
# canonical reproduce_output/steering dir.
OUT_ROOT="reproduce_output/steering"
for ((i = 0; i < ${#EXTRA[@]}; i++)); do
    if [ "${EXTRA[$i]}" = "--output" ] && [ $((i + 1)) -lt ${#EXTRA[@]} ]; then
        OUT_ROOT="${EXTRA[$((i + 1))]}"
    fi
done

N_GPU="$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)"

# Run one split of one condition, optionally pinned to a GPU.
run_split() {
    local cond="$1" split="$2" gpu="${3:-}"
    echo "[${cond}/${split}] GPU ${gpu:-<inherited>}"
    if [ -n "$gpu" ]; then
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" reproduce/run_steering.py \
            --condition "$cond" --splits "$split" "${EXTRA[@]+"${EXTRA[@]}"}"
    else
        "$PYTHON" reproduce/run_steering.py \
            --condition "$cond" --splits "$split" "${EXTRA[@]+"${EXTRA[@]}"}"
    fi
}

# Run one condition over both splits (2-GPU parallel when possible).
run_condition() {
    local cond="$1"
    local out="${OUT_ROOT}/${cond}"
    mkdir -p "$out"
    echo "=================================================================="
    echo "Steered generation: ${cond}"
    echo "=================================================================="
    if [ "${N_GPU}" -ge 2 ] && [ "${SERIAL:-0}" != "1" ]; then
        echo "  two splits on ${N_GPU} GPUs — test->GPU0, eval->GPU1 (logs: ${out}/step12_{test,eval}.log)"
        run_split "$cond" test 0 > "${out}/step12_test.log" 2>&1 &
        local pid_test=$!
        run_split "$cond" eval 1 > "${out}/step12_eval.log" 2>&1 &
        local pid_eval=$!
        wait "$pid_test" "$pid_eval"
    else
        for split in test eval; do
            run_split "$cond" "$split"
        done
    fi
}

for cond in "${CONDITIONS[@]}"; do
    run_condition "$cond"
done

echo "Done. Steered continuations under reproduce_output/steering/<condition>/<split>/<chorale>/<cut>/output.mid"

#!/usr/bin/env bash
#
# Step 3 — Extract activations for the per-layer patching experiment (phase 2).
#
# Runs Aria with activation hooks on the factual (original) and minorized
# (counterfactual) cuts, saving for each (chorale, cut, seed):
#   - the residual stream at the patch position (last prompt token), all layers
#   - logits/probs at the first 30 generated positions
# Output: reproduce_output/patching/<timestamp>_reproduce_{major,minor}/
#
# REQUIRES A GPU.
#
# Usage:
#     reproduce/extract_patching_activations.sh            # both major and minor
#     reproduce/extract_patching_activations.sh major      # only major
#     reproduce/extract_patching_activations.sh minor      # only minor
#
# The python interpreter defaults to whatever `python` is on PATH; activate the
# aria conda env first, or override with:  PYTHON=/path/to/python reproduce/...

set -euo pipefail

# Resolve repo root as the parent of this script's directory, then run from there
# so the relative data/output paths below are correct regardless of cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"

# Modes: use the args passed (e.g. "major"), or default to both.
if [ "$#" -gt 0 ]; then
    MODES=("$@")
else
    MODES=(major minor)
fi

for mode in "${MODES[@]}"; do
    echo "=================================================================="
    echo "Extracting activations for: ${mode}"
    echo "=================================================================="
    "$PYTHON" reproduce/extract_patching_activations.py \
        --name "reproduce_${mode}" \
        --data_dir "reproduce_output/data/${mode}_chorale_corpus" \
        --hooks
done

echo "Done. Outputs under reproduce_output/patching/"

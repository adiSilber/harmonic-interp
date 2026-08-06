#!/usr/bin/env bash
#
# Step 18 — Single-layer steering: is one layer enough?
#
# Re-runs the `parallel` and `relative` steering conditions with the direction added at
# ONE layer only (layer 12) instead of the full 11-15 range, at a correspondingly larger
# alpha, and re-scores them with the same Step-13/15/16 scripts. This is the experiment
# behind the paper's "a single layer carries most of the parallel-minor effect, but
# relative minor needs the whole range" claim.
#
# Everything lands in a parallel output root so the canonical Step-12 runs are untouched:
#
#     reproduce_output/steering_l12/<condition>/<split>/<chorale>/<cut>/output.mid
#     reproduce_output/steering_l12/resolutions.csv        <- resolution labels
#     reproduce_output/steering_l12/quality_metrics.csv    <- Table-1 metrics
#
# The FMD sets are written NEXT TO the main ones as `parallel_l12/` and `relative_l12/`
# (build_fmd_cont_data.py --set_suffix), so they are scored against the same bach/
# reference as the full-range runs and the numbers are directly comparable. The scores
# are merged into the shared fmd_cont_data/fmd_cont_scores.json.
#
# REQUIRES A GPU (generation + FMD). Steps 12 and 14-15 must have run first.
#
# Usage:
#     reproduce/run_single_layer_steering.sh              # layer 12, alpha 0.40
#     LAYER=13 ALPHA=0.40 reproduce/run_single_layer_steering.sh
#
# ALPHA is larger than the full-range alpha on purpose: 0.10 added at each of five
# layers is a much bigger total intervention than 0.10 added at one.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
LAYER="${LAYER:-12}"
ALPHA="${ALPHA:-0.40}"
SUFFIX="_l${LAYER}"
OUT="reproduce_output/steering_l${LAYER}"
CONDITIONS=(parallel relative)

echo "=== single-layer steering: layer ${LAYER}, alpha ${ALPHA} -> ${OUT} ==="

# 1. Generate (both splits, both GPUs — run_steering.sh handles the placement).
reproduce/run_steering.sh "${CONDITIONS[@]}" \
    --layers "$LAYER" --alpha "$ALPHA" --output "$OUT"

# 2. Label the first-chord resolution of every continuation.
"$PYTHON" reproduce/label_resolutions.py \
    --steering "$OUT" --conditions "${CONDITIONS[@]}" --out "$OUT/resolutions.csv"

# 3. FMD sets beside the main ones, sharing the same bach/ reference, then score them.
"$PYTHON" reproduce/build_fmd_cont_data.py \
    --steering "$OUT" --conditions "${CONDITIONS[@]}" --set_suffix "$SUFFIX"
"$PYTHON" reproduce/compute_fmd.py \
    --conditions "parallel${SUFFIX}" "relative${SUFFIX}" \
    --out "reproduce_output/fmd_cont_data/fmd_scores${SUFFIX}.json"

# Merge the new scores into the shared fmd_cont_scores.json (same reference + extractor,
# so the merged file is what one Step-15 run over all the sets would have produced).
"$PYTHON" - "$SUFFIX" <<'PY'
import json, sys
suffix = sys.argv[1]
main_path = "reproduce_output/fmd_cont_data/fmd_cont_scores.json"
main = json.load(open(main_path))
new = json.load(open(f"reproduce_output/fmd_cont_data/fmd_scores{suffix}.json"))
meta = ("reference", "n_reference", "model", "estimator")
if any(main[k] != new[k] for k in meta):
    raise SystemExit(f"refusing to merge: {main_path} and the new scores disagree on {meta}")
main["scores"].update(new["scores"])
json.dump(main, open(main_path, "w"), indent=2)
print("merged ->", main_path, sorted(main["scores"]))
PY

# 4. Table-1 metrics, reading the *_l<LAYER> FMD entries.
"$PYTHON" reproduce/compute_quality_metrics.py \
    --steering "$OUT" --conditions "${CONDITIONS[@]}" \
    --fmd_suffix "$SUFFIX" --out "$OUT/quality_metrics.csv"

# 5. Figure + the numbers for the paper text (CPU).
if [ "$LAYER" = "12" ]; then
    "$PYTHON" reproduce/plot_single_layer.py
fi

echo
echo "Done. Compare against the full-range runs:"
echo "  resolutions:  reproduce_output/steering/resolutions.csv     vs  $OUT/resolutions.csv"
echo "  quality:      reproduce_output/steering/quality_metrics.csv vs  $OUT/quality_metrics.csv"

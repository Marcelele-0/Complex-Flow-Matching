#!/usr/bin/env bash
# One arm of Tables 2/3, train then evaluate. Everything except the loss is
# identical to scripts/paper/table2_unet64.sh and table3_unet_sizes.sh.
#
#   LOSS_MODE=l2u bash scripts/paper/run_arm.sh NAME SIDE MANIFOLD COUPLING SEED [EPOCHS]
#
# LOSS_MODE:
#   l2u  (default) the fair loss: L2 in both geometries, unweighted phase term
#        (see scripts/paper/ablation_loss32.sh)
#   l1w  the arXiv v1 loss: L1 in both geometries, amplitude-weighted phase term
#   l1u  L1 in both geometries, unweighted phase term -- the cylinder's
#        counterpart of Cartesian L1, which has no phase weighting to remove
#
# Runs from the repository root through uv; nothing cluster-specific is assumed.
set -u
cd "$(dirname "$0")/../.."
NAME="$1"
SIDE="$2"
M="$3"
C="$4"
SEED="$5"
EPOCHS="${6:-40}"
LOSS_MODE="${LOSS_MODE:-l2u}"

# Everything but geometry, coupling, size and seed lives in conf/experiment/.
# The archived runs were launched with the equivalent explicit overrides;
# tests/test_paper_results.py checks that both compose to the same config.
case "$LOSS_MODE" in
  l2u) EXPERIMENT=paper_unet ;;
  l1u) EXPERIMENT=paper_unet_l1 ;;
  l1w) EXPERIMENT=paper_unet_v1loss ;;
  *) echo "unknown LOSS_MODE=$LOSS_MODE (expected l2u, l1u or l1w)"; exit 2 ;;
esac
COMMON=(
  +experiment="$EXPERIMENT" manifold="$M" training.coupling="$C"
  dataset.crop_size="[$SIDE,$SIDE]"
)

echo "[$(date '+%H:%M:%S')] train $NAME ($EXPERIMENT)"
uv run --no-sync python -m cfm.train "${COMMON[@]}" \
  training.epochs="$EPOCHS" training.seed="$SEED" logging.experiment_name="$NAME"
code=$?
if [ "$code" -ne 0 ]; then echo "FAILED train $NAME exit=$code"; exit "$code"; fi

echo "[$(date '+%H:%M:%S')] evaluate $NAME"
uv run --no-sync python -m cfm.evaluate "${COMMON[@]}" \
  evaluate.run_name="$NAME" evaluate.num_fields=64 evaluate.seed="$SEED" \
  evaluate.nfe='[1,2,4,8,100]' logging.experiment_name="${NAME}_eval"
code=$?
if [ "$code" -ne 0 ]; then echo "FAILED evaluate $NAME exit=$code"; exit "$code"; fi
echo "[$(date '+%H:%M:%S')] done $NAME"

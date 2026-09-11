#!/usr/bin/env bash
# One arm of Tables 2/3, train then evaluate. Everything except the loss is
# identical to scripts/paper/table2_unet64.sh and table3_unet_sizes.sh.
#
#   LOSS_MODE=l2u bash scripts/wcss/run_arm.sh NAME SIDE MANIFOLD COUPLING SEED [EPOCHS]
#
# LOSS_MODE:
#   l2u  (default) the fair loss: L2 in both geometries, unweighted phase term
#        (see scripts/paper/ablation_loss32.sh)
#   l1w  the arXiv v1 loss: L1 in both geometries, amplitude-weighted phase term
#
# Expects scripts/wcss/env.sh to have been sourced (cwd = checkout, uv on PATH).
set -u
NAME="$1"
SIDE="$2"
M="$3"
C="$4"
SEED="$5"
EPOCHS="${6:-40}"
LOSS_MODE="${LOSS_MODE:-l2u}"

# Cylindrical keys are ignored by the Euclidean manifold and vice versa, so one
# list serves both geometries.
case "$LOSS_MODE" in
  l2u) L=l2; W=false ;;
  l1w) L=l1; W=true ;;
  *) echo "unknown LOSS_MODE=$LOSS_MODE (expected l2u or l1w)"; exit 2 ;;
esac
LOSS=(
  training.loss.amp_loss_type="$L"
  training.loss.phase_loss_type="$L"
  training.loss.phase_amplitude_weighting="$W"
  training.loss.vel_loss_type="$L"
)
COMMON=(
  dataset=cylinder_toy_field model=c_unet manifold="$M"
  manifold.spatial_correlation=null
  training.bridge=noise training.coupling="$C"
  dataset.crop_size="[$SIDE,$SIDE]" dataset.coupling=0.5
)

echo "[$(date '+%H:%M:%S')] train $NAME"
uv run --no-sync python -m cfm.train "${COMMON[@]}" "${LOSS[@]}" \
  training.epochs="$EPOCHS" training.batch_size=64 \
  training.grad_clip=null training.compile=false training.seed="$SEED" \
  dataset.size=8192 \
  logging.use_wandb=false logging.experiment_name="$NAME"
code=$?
if [ "$code" -ne 0 ]; then echo "FAILED train $NAME exit=$code"; exit "$code"; fi

echo "[$(date '+%H:%M:%S')] evaluate $NAME"
uv run --no-sync python -m cfm.evaluate "${COMMON[@]}" "${LOSS[@]}" \
  evaluate.run_name="$NAME" evaluate.num_fields=64 evaluate.seed="$SEED" \
  evaluate.nfe='[1,2,4,8,100]' \
  logging.use_wandb=false logging.experiment_name="${NAME}_eval"
code=$?
if [ "$code" -ne 0 ]; then echo "FAILED evaluate $NAME exit=$code"; exit "$code"; fi
echo "[$(date '+%H:%M:%S')] done $NAME"

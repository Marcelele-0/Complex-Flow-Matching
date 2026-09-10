#!/usr/bin/env bash
# Does concentrating the phase prior restore the cylinder's angular learning?
#
# Cylindrical arm only, independent coupling, everything else identical to the
# Table 1 grid. This is a WITHIN-ARM sweep: changing the prior means the two
# geometries no longer share one, so these numbers answer "does the cylinder
# recover", not "does the cylinder now beat the plane".
set -u
cd "$(dirname "$0")/../.."
LOG="$1"; mkdir -p "$LOG"
stamp() { date '+%H:%M:%S'; }
run() {
  local label="$1"; shift
  "$@" > "$LOG/${label}.log" 2>&1
  local status=$?
  echo "[$(stamp)] ${label} exit=${status}"
  if [ "$status" -ne 0 ]; then echo "FAILED: ${label}"; tail -5 "$LOG/${label}.log"; exit 1; fi
}

for SPREAD in 0.25 0.5 1.0 2.0 null; do
  for SEED in 0 1; do
    TAG=$(echo "$SPREAD" | tr '.' 'p')
    NAME="pp_${TAG}_s${SEED}"
    echo "[$(stamp)] === phase_spread=$SPREAD seed=$SEED ==="
    run "train_${NAME}" uv run python -m cfm.train \
      dataset=cylinder_toy_iid model=mlp_gate manifold=cylindrical \
      manifold.phase_spread="$SPREAD" \
      training.bridge=noise training.coupling=independent \
      training.epochs=40 training.batch_size=64 \
      training.grad_clip=null training.compile=false training.seed="$SEED" \
      dataset.size=8192 dataset.crop_size='[64,64]' dataset.coupling=0.5 \
      logging.use_wandb=false logging.experiment_name="$NAME"
    run "eval_${NAME}" uv run python -m cfm.evaluate \
      dataset=cylinder_toy_iid model=mlp_gate manifold=cylindrical \
      manifold.phase_spread="$SPREAD" \
      training.bridge=noise dataset.crop_size='[64,64]' dataset.coupling=0.5 \
      evaluate.run_name="$NAME" evaluate.num_fields=64 evaluate.seed="$SEED" \
      evaluate.nfe='[1,2,4,8,100]' \
      logging.use_wandb=false logging.experiment_name="${NAME}_eval"
  done
done
echo "[$(stamp)] done"

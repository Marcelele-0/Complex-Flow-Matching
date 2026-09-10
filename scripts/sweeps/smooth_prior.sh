#!/usr/bin/env bash
# Does a spatially correlated prior rescue the coupling -- and with it, phase?
#
# Smooth data (cylinder_toy_field, correlation length 4), 64x64, rho 0.5.
# Full 2x2 of geometry x coupling, crossed with the prior's correlation length:
# null (white), 4 (matched to the data), 16 (smoother than the data). The prior
# is sample-matched across geometries, so it is not a confound. Architecture,
# epochs, batch and learning rate are identical to Table 1.
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

for SC in null 4 16; do
  for M in cylindrical euclidean; do
    for C in independent ot; do
      for SEED in 0 1; do
        NAME="sp_${M}_${C}_sc${SC}_s${SEED}"
        echo "[$(stamp)] === $NAME ==="
        run "train_${NAME}" uv run python -m cfm.train \
          dataset=cylinder_toy_field model=mlp_gate manifold="$M" \
          manifold.spatial_correlation="$SC" \
          training.bridge=noise training.coupling="$C" \
          training.epochs=40 training.batch_size=64 \
          training.grad_clip=null training.compile=false training.seed="$SEED" \
          dataset.size=8192 dataset.crop_size='[64,64]' dataset.coupling=0.5 \
          logging.use_wandb=false logging.experiment_name="$NAME"
        run "eval_${NAME}" uv run python -m cfm.evaluate \
          dataset=cylinder_toy_field model=mlp_gate manifold="$M" \
          manifold.spatial_correlation="$SC" \
          training.bridge=noise training.coupling="$C" dataset.crop_size='[64,64]' dataset.coupling=0.5 \
          evaluate.run_name="$NAME" evaluate.num_fields=64 evaluate.seed="$SEED" \
          evaluate.nfe='[1,2,4,8,100]' \
          logging.use_wandb=false logging.experiment_name="${NAME}_eval"
      done
    done
  done
done
echo "[$(stamp)] done"

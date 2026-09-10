#!/usr/bin/env bash
# The white-prior U-Net comparison of unet_prior.sh, at 32x32 and 16x16.
#
# Same data law, same correlation length in pixels (4), same model, epochs,
# batch and learning rate; only the field size changes. This is what says how
# the cylinder's few-step advantage and the value of OT move with dimension, and
# so whether modelling patches rather than whole fields is worth trying.
# c_unet needs sides divisible by 16.
set -u
cd "$(dirname "$0")/../.."
LOG="$1"; mkdir -p "$LOG"
stamp() { date '+%H:%M:%S'; }

# $? must be captured before any command substitution: $(stamp) runs date,
# which overwrites it.
run() {
  local label="$1"; shift
  "$@" > "$LOG/${label}.log" 2>&1
  local code=$?
  echo "[$(stamp)] ${label} exit=${code}"
  if [ "$code" -ne 0 ]; then echo "FAILED: ${label}"; tail -5 "$LOG/${label}.log"; exit 1; fi
}

for SIDE in 32 16; do
  for M in cylindrical euclidean; do
    for C in independent ot; do
      for SEED in 0 1; do
        NAME="unsz_${M}_${C}_${SIDE}_s${SEED}"
        echo "[$(stamp)] === $NAME ==="
        run "train_${NAME}" uv run python -m cfm.train \
          dataset=cylinder_toy_field model=c_unet manifold="$M" \
          training.bridge=noise training.coupling="$C" \
          training.epochs=40 training.batch_size=64 \
          training.grad_clip=null training.compile=false training.seed="$SEED" \
          dataset.size=8192 dataset.crop_size="[$SIDE,$SIDE]" dataset.coupling=0.5 \
          logging.use_wandb=false logging.experiment_name="$NAME"
        run "eval_${NAME}" uv run python -m cfm.evaluate \
          dataset=cylinder_toy_field model=c_unet manifold="$M" \
          training.bridge=noise training.coupling="$C" \
          dataset.crop_size="[$SIDE,$SIDE]" dataset.coupling=0.5 \
          evaluate.run_name="$NAME" evaluate.num_fields=64 evaluate.seed="$SEED" \
          evaluate.nfe='[1,2,4,8,100]' \
          logging.use_wandb=false logging.experiment_name="${NAME}_eval"
      done
    done
  done
done
echo "[$(stamp)] done"

#!/usr/bin/env bash
# The smooth-prior sweep again, with a model that mixes space.
#
# A pointwise MLP can only pass a prior's spatial correlation through; it cannot
# create structure from white noise, and it cannot use a neighbour to infer a
# coefficient's endpoint. A U-Net can do both, for both geometries -- so the
# question is whether spatial context helps the cylinder more than the plane.
# The white prior is kept on purpose: it is the only arm that tests whether the
# network learns spatial structure rather than inheriting it.
#
# c_unet (3.6M parameters; 576 more on the cylinder for its third input channel,
# 0.016%). Data, epochs, batch and learning rate as in the MLP sweep.
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

for SC in 4 null; do
  for M in cylindrical euclidean; do
    for C in independent ot; do
      for SEED in 0 1; do
        NAME="un_${M}_${C}_sc${SC}_s${SEED}"
        echo "[$(stamp)] === $NAME ==="
        run "train_${NAME}" uv run python -m cfm.train \
          dataset=cylinder_toy_field model=c_unet manifold="$M" \
          manifold.spatial_correlation="$SC" \
          training.bridge=noise training.coupling="$C" \
          training.epochs=40 training.batch_size=64 \
          training.grad_clip=null training.compile=false training.seed="$SEED" \
          dataset.size=8192 dataset.crop_size='[64,64]' dataset.coupling=0.5 \
          logging.use_wandb=false logging.experiment_name="$NAME"
        run "eval_${NAME}" uv run python -m cfm.evaluate \
          dataset=cylinder_toy_field model=c_unet manifold="$M" \
          manifold.spatial_correlation="$SC" \
          training.bridge=noise training.coupling="$C" \
          dataset.crop_size='[64,64]' dataset.coupling=0.5 \
          evaluate.run_name="$NAME" evaluate.num_fields=64 evaluate.seed="$SEED" \
          evaluate.nfe='[1,2,4,8,100]' \
          logging.use_wandb=false logging.experiment_name="${NAME}_eval"
      done
    done
  done
done
echo "[$(stamp)] done"

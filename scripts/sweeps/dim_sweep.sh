#!/usr/bin/env bash
# Does the cylinder's few-step advantage survive at field scale?
#
# Independent coupling, cylinder_toy_iid, fields of 16, 32 and 64 pixels a side,
# both geometries, two seeds. grad_clip is null because the training config
# warns it is not scale-invariant and the two geometries' losses differ ~4x.
set -u
cd "$(dirname "$0")/../.."
LOG="$1"; mkdir -p "$LOG"
stamp() { date '+%H:%M:%S'; }

# $? must be captured before any command substitution: $(stamp) runs date,
# which overwrites it.
run() {
  local label="$1"; shift
  "$@" > "$LOG/${label}.log" 2>&1
  local status=$?
  echo "[$(stamp)] ${label} exit=${status}"
  if [ "$status" -ne 0 ]; then echo "FAILED: ${label}"; tail -5 "$LOG/${label}.log"; exit 1; fi
}

for SIDE in 16 32 64; do
  for M in cylindrical euclidean; do
    for SEED in 0 1; do
      NAME="dim_${M}_${SIDE}_s${SEED}"
      echo "[$(stamp)] === $NAME ==="
      run "train_${NAME}" uv run python -m cfm.train \
        dataset=cylinder_toy_iid model=mlp_gate manifold="$M" \
        training.bridge=noise training.epochs=40 training.batch_size=64 \
        training.grad_clip=null training.compile=false training.seed="$SEED" \
        dataset.size=8192 dataset.crop_size="[$SIDE,$SIDE]" dataset.coupling=0.5 \
        logging.use_wandb=false logging.experiment_name="$NAME"
      run "eval_${NAME}" uv run python -m cfm.evaluate \
        dataset=cylinder_toy_iid model=mlp_gate manifold="$M" \
        training.bridge=noise dataset.crop_size="[$SIDE,$SIDE]" dataset.coupling=0.5 \
        evaluate.run_name="$NAME" evaluate.num_fields=64 evaluate.seed="$SEED" \
        logging.use_wandb=false logging.experiment_name="${NAME}_eval"
    done
  done
done
echo "[$(stamp)] done"

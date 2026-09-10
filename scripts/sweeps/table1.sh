#!/usr/bin/env bash
# Table 1: geometry x coupling, unconditional synthesis on 64x64 complex fields.
# Identical architecture, epochs, batch and learning rate across all four arms.
# grad_clip is disabled because the training config warns it is not scale
# invariant and the two geometries' losses differ ~4x in magnitude.
set -u
cd "$(dirname "$0")/../.."
LOG="$1"; mkdir -p "$LOG"
stamp() { date '+%H:%M:%S'; }

# $? must be captured before any command substitution: $(stamp) runs date, which
# overwrites it. That masked eight failed runs on the previous attempt.
run() {
  local label="$1"; shift
  "$@" > "$LOG/${label}.log" 2>&1
  local status=$?
  echo "[$(stamp)] ${label} exit=${status}"
  if [ "$status" -ne 0 ]; then
    echo "FAILED: ${label} -- see $LOG/${label}.log"
    tail -5 "$LOG/${label}.log"
    exit 1
  fi
}

for M in euclidean cylindrical; do
  for C in independent ot; do
    for SEED in 0 1; do
      NAME="t1_${M}_${C}_s${SEED}"
      echo "[$(stamp)] === $NAME ==="
      run "train_${NAME}" uv run python -m cfm.train \
        dataset=cylinder_toy_iid model=mlp_gate manifold="$M" \
        training.bridge=noise training.coupling="$C" \
        training.epochs=40 training.batch_size=64 \
        training.grad_clip=null training.compile=false training.seed="$SEED" \
        dataset.size=8192 dataset.crop_size='[64,64]' dataset.coupling=0.5 \
        logging.use_wandb=false logging.experiment_name="$NAME"
      run "eval_${NAME}" uv run python -m cfm.evaluate \
        dataset=cylinder_toy_iid model=mlp_gate manifold="$M" \
        training.bridge=noise training.coupling="$C" dataset.crop_size='[64,64]' dataset.coupling=0.5 \
        evaluate.run_name="$NAME" evaluate.num_fields=64 evaluate.seed="$SEED" \
        evaluate.nfe='[1,2,4,8,100]' \
        logging.use_wandb=false logging.experiment_name="${NAME}_eval"
    done
  done
done
echo "[$(stamp)] done"

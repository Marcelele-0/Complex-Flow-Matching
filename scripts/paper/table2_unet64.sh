#!/usr/bin/env bash
# Table 2 of the paper: U-Net on 64x64 synthetic complex fields,
# {Cartesian, cylindrical} x {independent, joint OT} x 2 seeds = 8 runs.
#
#   bash scripts/paper/table2_unet64.sh LOGDIR
#
# Each run trains c_unet for 40 epochs on cylinder_toy_field (rho 0.5, correlation
# length 4, 8192 fields, batch 64, white uniform-disc prior) and evaluates it at
# k = 1, 2, 4, 8, 100 Heun steps on 64 fields. About 20 minutes per run on an RTX 4070
# Ti SUPER, ~2.7 h in total. Then prints the table with scripts/paper/paper_tables.py.
#
# Run names match the archived evaluations (docs/reproduce/paper_results/unet_eval_metrics.json),
# so `paper_tables.py --archive` prints the paper's numbers for comparison. Retraining
# is seeded but GPU kernels are not bitwise deterministic: expect agreement within the
# seed spread, not to the digit. Re-evaluating a fixed checkpoint is exact.
#
# Environment overrides for a quick end-to-end check (not the paper's numbers):
#   EPOCHS (40)  SIZE (8192)  FIELDS (64)  SEEDS ("0 1")
#   PREFIX ("")  -- e.g. PREFIX=smoke_ so a short run never shadows a real one;
#                   read it back with `paper_tables.py --prefix smoke_`.
set -u
cd "$(dirname "$0")/../.."
LOG="${1:?usage: $0 LOGDIR}"
mkdir -p "$LOG"
EPOCHS="${EPOCHS:-40}"
SIZE="${SIZE:-8192}"
FIELDS="${FIELDS:-64}"
SEEDS="${SEEDS:-0 1}"
PREFIX="${PREFIX:-}"

stamp() { date '+%H:%M:%S'; }
run() {
  local label="$1"; shift
  "$@" > "$LOG/${label}.log" 2>&1
  local code=$?
  echo "[$(stamp)] ${label} exit=${code}"
  if [ "$code" -ne 0 ]; then echo "FAILED: ${label}"; tail -5 "$LOG/${label}.log"; exit 1; fi
}

for M in cylindrical euclidean; do
  for C in independent ot; do
    for SEED in $SEEDS; do
      NAME="${PREFIX}un_${M}_${C}_scnull_s${SEED}"
      echo "[$(stamp)] === $NAME ==="
      run "train_${NAME}" uv run python -m cfm.train \
        dataset=cylinder_toy_field model=c_unet manifold="$M" \
        manifold.spatial_correlation=null \
        training.bridge=noise training.coupling="$C" \
        training.epochs="$EPOCHS" training.batch_size=64 \
        training.grad_clip=null training.compile=false training.seed="$SEED" \
        dataset.size="$SIZE" dataset.crop_size='[64,64]' dataset.coupling=0.5 \
        logging.use_wandb=false logging.experiment_name="$NAME"
      run "eval_${NAME}" uv run python -m cfm.evaluate \
        dataset=cylinder_toy_field model=c_unet manifold="$M" \
        manifold.spatial_correlation=null \
        training.bridge=noise training.coupling="$C" \
        dataset.crop_size='[64,64]' dataset.coupling=0.5 \
        evaluate.run_name="$NAME" evaluate.num_fields="$FIELDS" evaluate.seed="$SEED" \
        evaluate.nfe='[1,2,4,8,100]' \
        logging.use_wandb=false logging.experiment_name="${NAME}_eval"
    done
  done
done
echo "[$(stamp)] done"
uv run python scripts/paper/paper_tables.py --prefix "$PREFIX"

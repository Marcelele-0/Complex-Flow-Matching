#!/usr/bin/env bash
# Table 3 of the paper: the same U-Net comparison at 16x16 and 32x32,
# {Cartesian, cylindrical} x {independent, joint OT} x 2 seeds x 2 sizes = 16 runs.
# The 64x64 rows come from scripts/paper/table2_unet64.sh; run that first (or read
# the 64x64 rows from the archive with `paper_tables.py --archive`).
#
#   bash scripts/paper/table3_unet_sizes.sh LOGDIR
#
# Same data, model and schedule as Table 2, only the crop changes. About 2-3 minutes
# per run on an RTX 4070 Ti SUPER, ~40 min in total. Then prints Tables 2 and 3 and the
# per-seed checks of Section 5.3 (cylinder OT vs independent: 11-44%, separated in 7
# of 9 cells) with scripts/paper/paper_tables.py.
#
# Retraining is seeded but GPU kernels are not bitwise deterministic: expect agreement
# within the seed spread, not to the digit.
#
# Environment overrides for a quick end-to-end check (not the paper's numbers):
#   EPOCHS (40)  SIZE (8192)  FIELDS (64)  SEEDS ("0 1")  SIDES ("32 16")
#   PREFIX ("")  -- e.g. PREFIX=smoke_ so a short run never shadows a real one.
set -u
cd "$(dirname "$0")/../.."
LOG="${1:?usage: $0 LOGDIR}"
mkdir -p "$LOG"
EPOCHS="${EPOCHS:-40}"
SIZE="${SIZE:-8192}"
FIELDS="${FIELDS:-64}"
SEEDS="${SEEDS:-0 1}"
SIDES="${SIDES:-32 16}"
PREFIX="${PREFIX:-}"

stamp() { date '+%H:%M:%S'; }
run() {
  local label="$1"; shift
  "$@" > "$LOG/${label}.log" 2>&1
  local code=$?
  echo "[$(stamp)] ${label} exit=${code}"
  if [ "$code" -ne 0 ]; then echo "FAILED: ${label}"; tail -5 "$LOG/${label}.log"; exit 1; fi
}

for SIDE in $SIDES; do
  for M in cylindrical euclidean; do
    for C in independent ot; do
      for SEED in $SEEDS; do
        NAME="${PREFIX}unsz_${M}_${C}_${SIDE}_s${SEED}"
        echo "[$(stamp)] === $NAME ==="
        run "train_${NAME}" uv run python -m cfm.train \
          dataset=cylinder_toy_field model=c_unet manifold="$M" \
          training.bridge=noise training.coupling="$C" \
          training.epochs="$EPOCHS" training.batch_size=64 \
          training.grad_clip=null training.compile=false training.seed="$SEED" \
          dataset.size="$SIZE" dataset.crop_size="[$SIDE,$SIDE]" dataset.coupling=0.5 \
          logging.use_wandb=false logging.experiment_name="$NAME"
        run "eval_${NAME}" uv run python -m cfm.evaluate \
          dataset=cylinder_toy_field model=c_unet manifold="$M" \
          training.bridge=noise training.coupling="$C" \
          dataset.crop_size="[$SIDE,$SIDE]" dataset.coupling=0.5 \
          evaluate.run_name="$NAME" evaluate.num_fields="$FIELDS" evaluate.seed="$SEED" \
          evaluate.nfe='[1,2,4,8,100]' \
          logging.use_wandb=false logging.experiment_name="${NAME}_eval"
      done
    done
  done
done
echo "[$(stamp)] done"
uv run python scripts/paper/paper_tables.py --prefix "$PREFIX"

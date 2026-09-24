#!/bin/bash
# Re-score existing checkpoints with the current metrics, without retraining.
#
# A metric added after a cohort has run leaves the cohort's archives without it. This
# re-runs `cyfm.evaluate` against checkpoints that already exist, found by
# `evaluate.run_name`, and writes to a separate `logging.experiment_name` so the original
# evaluation is preserved rather than overwritten. Nothing here trains.
#
# It exists because `phase_lag_one` (circular coherence of adjacent phases) was added on
# 2026-09-18, after the knee cohort. Keeping the invocation in the repository rather than
# in a home directory is the difference between a number someone can reproduce from a
# checkout and a number they have to take on trust.
#
#   PREFIX     run-name prefix of the cohort to re-score, e.g. t5_ or t5w_
#   OUT_PREFIX prefix for the new evaluation names; must differ from PREFIX
#   KSPACE_CROP  pass 64 for a cohort trained on the cropped matrix, empty otherwise
#   EXTRA_ARGS   further Hydra overrides, e.g. the loss the cohort was trained with
#   SEEDS      default "0", widen for the full grid
#
# What produced the re-scored numbers of 2026-09-18:
#
#   PREFIX=t5_    OUT_PREFIX=ps2_ sbatch scripts/cluster/rescore_metrics.sbatch
#   PREFIX=t5c64_ OUT_PREFIX=ps2_ KSPACE_CROP=64 sbatch scripts/cluster/rescore_metrics.sbatch
#   PREFIX=t5w_   OUT_PREFIX=ps3_ EXTRA_ARGS=training.loss.phase_amplitude_weighting=true \
#     sbatch scripts/cluster/rescore_metrics.sbatch
#   PREFIX=t5wc64_ OUT_PREFIX=ps3_ KSPACE_CROP=64 \
#     EXTRA_ARGS=training.loss.phase_amplitude_weighting=true \
#     sbatch scripts/cluster/rescore_metrics.sbatch
#
# The loss override matters even for scoring: run_arm.sh records the protocol on every row,
# and a row whose recorded loss differs from the checkpoint's is a row that cannot be
# checked against the others.
#
# Budget: eight evaluations of 64 fields over the [1,2,4,8,100] grid, measured 5 minutes on
# one H100. Reading val.h5 from lustre rather than staging it is deliberate: 64 fields is a
# few hundred reads, not 40 epochs of them.
#SBATCH --job-name=cyfm-rescore
#SBATCH --partition=lem-gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=/path/to/project/CyFM/logs/%x-%j.txt
#SBATCH --error=/path/to/project/CyFM/logs/%x-%j.err
set -u
source /path/to/project/CyFM/scripts/cluster/env.sh

PREFIX="${PREFIX:?set PREFIX to the cohort to re-score, e.g. t5_}"
OUT_PREFIX="${OUT_PREFIX:?set OUT_PREFIX for the new evaluation names}"
KSPACE_CROP="${KSPACE_CROP:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SEEDS="${SEEDS:-0}"
GEOMS="${GEOMS:-cylindrical euclidean}"
COUPLINGS="${COUPLINGS:-ot independent}"
NFE="${NFE:-[1,2,4,8,100]}"
STORE_DIR="${STORE_DIR:-$CYFM_ROOT/data/knee_pd}"

[ "$PREFIX" != "$OUT_PREFIX" ] || { echo "OUT_PREFIX must differ from PREFIX"; exit 2; }

CROP_OVERRIDE=()
if [ -n "$KSPACE_CROP" ]; then
  CROP_OVERRIDE=(dataset.kspace_crop="$KSPACE_CROP")
fi
EXTRA_OVERRIDE=()
if [ -n "$EXTRA_ARGS" ]; then
  # Word splitting is wanted: EXTRA_ARGS may carry several overrides.
  # shellcheck disable=SC2206
  EXTRA_OVERRIDE=($EXTRA_ARGS)
fi

failed=0
for M in $GEOMS; do
  for C in $COUPLINGS; do
    for SEED in $SEEDS; do
      NAME="${PREFIX}${M}_${C}_s${SEED}"
      echo "=== re-scoring $NAME (crop=${KSPACE_CROP:-none}) ==="
      uv run --no-sync python -m cyfm.evaluate +experiment=table5_fastmri \
        manifold="$M" training.coupling="$C" \
        dataset.data_dir="$STORE_DIR" dataset.store=val.h5 dataset.role=all \
        ${CROP_OVERRIDE[@]+"${CROP_OVERRIDE[@]}"} \
        ${EXTRA_OVERRIDE[@]+"${EXTRA_OVERRIDE[@]}"} \
        evaluate.run_name="$NAME" evaluate.num_fields=64 evaluate.seed=0 \
        evaluate.nfe="$NFE" \
        logging.experiment_name="${OUT_PREFIX}${NAME}_eval" || {
          echo "FAILED $NAME"
          failed=$((failed + 1))
        }
    done
  done
done

echo "done, $failed failed"
exit $((failed > 0))

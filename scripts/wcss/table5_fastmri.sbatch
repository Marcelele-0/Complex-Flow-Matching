#!/usr/bin/env bash
# Table 5: fastMRI knee CORPD at 320x320, one array task per run.
#   {cylindrical, euclidean} x {independent, ot} x seeds {0..4}   20 tasks
#   complex_diffusion x independent x seeds {0..4}                 5 tasks, scored twice
#                                                                 -- = 25 tasks, 30 evaluations
#
#   sbatch scripts/wcss/table5_fastmri.sbatch
#
# Prerequisite, and it does not happen on its own when the intake array drains:
#
#   uv run python scripts/data/build_knee_pd_store.py \
#     --merge data/knee_pd/parts/train/*.h5 --out data/knee_pd/train.h5 --split train
#
# The split is fastMRI's own. Its train and val cohorts are patient-disjoint by
# construction, so training reads train.h5 and the evaluation scores against val.h5
# with role=all, rather than hashing one store into two roles. That is also the
# protocol the score-based-prior literature reports, which is what lets our numbers
# sit beside theirs.
#
# The diffusion baseline (#83) is in this grid: it trains once per seed and is scored
# twice, at the flow arms' matched budget and in its own many-step regime, which is
# a second evaluation of one checkpoint rather than a second training run. Only the
# coupling is meaningless for it -- a score model has no pairing -- so it runs at
# `independent` alone, giving 5 tasks rather than 10. The phase-free floor (#89) is
# still out: it is a separate arm and a separate issue.
#
# sigma_max is the VE prior's width and must match the data: ~140 at 320x320 for
# peak-normalised knee fields. Confirm it against the store before launching, with
# cfm.manifolds.complex_diffusion.calibrate_sigma_max; a value from another
# resolution is wrong in one direction or the other.
#
# The many-step scoring is a regime, not a column: the matched-NFE columns top out at
# 199 model calls, and a score-based sampler is not used there. It is therefore swept
# at its own single point, NATIVE_STEPS below, and occupies the last column of the
# table alone. Sweeping it over [1,2,4,8,100] instead -- as this script first did --
# produces 99 steps against 100 in the top column, which is the same sampler twice.
#
# Budget: 40 epochs at ~54 s/epoch measured by the #76 gate, so ~36 min and 0.6 GPU-h
# per task, ~15 GPU-h for the twenty-five. On top of that the five diffusion tasks pay
# one many-step scoring each: NATIVE_STEPS * (1 + corrector_steps) model calls per
# batch, ~2000 at the default, which is the reason the wall clock below is 2 h rather
# than the 1.5 h the flow arms need. MEASURE the first diffusion cell before releasing
# the other four -- that estimate is arithmetic, not a measurement.
#SBATCH --job-name=cyfm-table5
#SBATCH --partition=lem-gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:hopper:1
#SBATCH --array=0-24
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/CyFM/logs/%x-%A_%a.txt
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/CyFM/logs/%x-%A_%a.err
set -u
source /lustre/pd03/hpc-danbor2008-1756464546/CyFM/scripts/wcss/env.sh

PREFIX="${PREFIX:-t5_}"
GEOMS="${GEOMS:-cylindrical euclidean}"
COUPLINGS="${COUPLINGS:-independent ot}"
SEEDS="${SEEDS:-0 1 2 3 4}"
STORE_DIR="${STORE_DIR:-$CYFM_ROOT/data/knee_pd}"

DIFFUSION="${DIFFUSION:-complex_diffusion}"
SIGMA_MAX="${SIGMA_MAX:-140.0}"
# Must equal TABLE5_NATIVE_STEPS in scripts/paper/latex_tables.py: the table reads
# this row's cell by its step count, so a value only changed here yields a row the
# renderer cannot find.
NATIVE_STEPS="${NATIVE_STEPS:-1000}"

TASKS=()
for M in $GEOMS; do
  for C in $COUPLINGS; do
    for SEED in $SEEDS; do
      TASKS+=("$M $C $SEED")
    done
  done
done
# The diffusion arm, at independent pairing only: a score model has no coupling
# to vary, so a second column would be the same run under a different name.
if [ -n "$DIFFUSION" ]; then
  for SEED in $SEEDS; do
    TASKS+=("$DIFFUSION independent $SEED")
  done
fi
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#TASKS[@]}" ]; then
  echo "task $SLURM_ARRAY_TASK_ID out of range: grid has ${#TASKS[@]} tasks"
  exit 2
fi
read -r M C SEED <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
# The diffusion tasks do not use this name -- they produce a matched and a native
# run below -- so it is only computed for the flow arms. Echoing a name no
# directory will ever carry sends anyone grepping the logs for a failed cell to
# a path that does not exist.
if [ "$M" != "$DIFFUSION" ]; then
  NAME="${PREFIX}${M}_${C}_s${SEED}"
else
  NAME="${PREFIX}${DIFFUSION}_{heun,native}_independent_s${SEED}"
fi

for f in train.h5 train.json val.h5 val.json; do
  [ -f "$STORE_DIR/$f" ] || { echo "missing $STORE_DIR/$f; merge the parts first (see header)"; exit 2; }
done

# GPU nodes carry 7 TB of node-local disk, unlike the CPU partition where TmpDisk=0.
# Training makes 40 epochs of random reads over 5324 slices, so the stores are staged
# here once rather than served from lustre on every step. Roughly 5 GB and seconds.
STAGE="${TMPDIR:-/tmp/$USER}/table5-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE"
t0=$(date +%s)
cp "$STORE_DIR"/train.h5 "$STORE_DIR"/train.json "$STORE_DIR"/val.h5 "$STORE_DIR"/val.json "$STAGE/"
echo "staged $(du -sh "$STAGE" | cut -f1) in $(( $(date +%s) - t0 ))s to $STAGE"

echo "task=$SLURM_ARRAY_TASK_ID host=$(hostname) name=$NAME"

COMMON_ENV=(
  EXPERIMENT=table5_fastmri
  DATA_DIR="$STAGE"
  TRAIN_STORE=train.h5
  EVAL_STORE=val.h5
  HOLDOUT_ROLE=all
)

if [ "$M" != "$DIFFUSION" ]; then
  env "${COMMON_ENV[@]}" bash scripts/paper/run_arm.sh "$NAME" native "$M" "$C" "$SEED" 40
  exit $?
fi

# One training run, two scorings. The checkpoint is the same; only the step
# budget differs, so the many-step row must not retrain -- it would be a
# different model and the comparison would stop being between two regimes of one
# baseline. run_arm.sh trains then evaluates, so the matched pass produces the
# checkpoint and the native pass re-evaluates it.
MATCHED_NAME="${PREFIX}${DIFFUSION}_heun_independent_s${SEED}"
NATIVE_NAME="${PREFIX}${DIFFUSION}_native_independent_s${SEED}"
EXTRA=(manifold.sigma_max="$SIGMA_MAX")

env "${COMMON_ENV[@]}" EXTRA_OVERRIDES="manifold.nfe_mode=heun ${EXTRA[*]}" \
  bash scripts/paper/run_arm.sh "$MATCHED_NAME" native "$DIFFUSION" independent "$SEED" 40
code=$?
if [ "$code" -ne 0 ]; then echo "FAILED matched $MATCHED_NAME exit=$code"; exit "$code"; fi

echo "[$(date '+%H:%M:%S')] evaluate $NATIVE_NAME ($NATIVE_STEPS steps, same checkpoint)"
# training.coupling is set here even though a score model has no pairing: run_arm.sh
# always records it, and the archive's integrity test reads one protocol off every
# row. A row missing it is a row that cannot be checked against the others.
uv run --no-sync python -m cfm.evaluate +experiment=table5_fastmri \
  manifold="$DIFFUSION" manifold.nfe_mode=native "${EXTRA[@]}" \
  training.coupling=independent \
  dataset.data_dir="$STAGE" dataset.store=val.h5 dataset.role=all \
  evaluate.run_name="$MATCHED_NAME" evaluate.num_fields=64 evaluate.seed="$SEED" \
  evaluate.nfe="[$NATIVE_STEPS]" logging.experiment_name="${NATIVE_NAME}_eval"

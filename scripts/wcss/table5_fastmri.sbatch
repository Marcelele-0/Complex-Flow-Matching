#!/usr/bin/env bash
# Table 5: fastMRI knee CORPD at 320x320, one array task per run.
#   {cylindrical, euclidean} x {independent, ot} x seeds {0..4} = 20 tasks
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
# The two missing rows of the table, the diffusion baseline (#83) and the phase-free
# floor (#89), are not in this grid; they are separate arms and separate issues.
#
# Budget: 40 epochs at ~54 s/epoch measured by the #76 gate, so ~36 min and 0.6 GPU-h
# per task, ~12 GPU-h for the twenty. The wall clock below is set to twice the measured
# run so a slow node does not lose a cell.
#SBATCH --job-name=cyfm-table5
#SBATCH --partition=lem-gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:hopper:1
#SBATCH --array=0-19
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/CyFM/logs/%x-%A_%a.txt
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/CyFM/logs/%x-%A_%a.err
set -u
source /lustre/pd03/hpc-danbor2008-1756464546/CyFM/scripts/wcss/env.sh

PREFIX="${PREFIX:-t5_}"
GEOMS="${GEOMS:-cylindrical euclidean}"
COUPLINGS="${COUPLINGS:-independent ot}"
SEEDS="${SEEDS:-0 1 2 3 4}"
STORE_DIR="${STORE_DIR:-$CYFM_ROOT/data/knee_pd}"

TASKS=()
for M in $GEOMS; do
  for C in $COUPLINGS; do
    for SEED in $SEEDS; do
      TASKS+=("$M $C $SEED")
    done
  done
done
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#TASKS[@]}" ]; then
  echo "task $SLURM_ARRAY_TASK_ID out of range: grid has ${#TASKS[@]} tasks"
  exit 2
fi
read -r M C SEED <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
NAME="${PREFIX}${M}_${C}_s${SEED}"

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
EXPERIMENT=table5_fastmri \
DATA_DIR="$STAGE" \
TRAIN_STORE=train.h5 \
EVAL_STORE=val.h5 \
HOLDOUT_ROLE=all \
  bash scripts/paper/run_arm.sh "$NAME" native "$M" "$C" "$SEED" 40

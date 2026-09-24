#!/usr/bin/env bash
# Table 6: LibriSpeech complex STFT, one array task per run.
#   {cylindrical, euclidean} x {independent, ot} x seeds {0..4} = 20 tasks
#
#   sbatch scripts/cluster/table6_audio.sbatch
#
# Prerequisite, built once by scripts/cluster/build_stft_store.sbatch:
#   data/librispeech_stft/librispeech.h5 and its .json manifest
#
# The split differs from Table 5 and that is why this is a separate script. fastMRI
# ships patient-disjoint train and val cohorts, so that grid trains on one store and
# scores against another. LibriSpeech does not, so dev-clean and test-clean are pooled
# into one store and hashed by SPEAKER: training reads role=fit, evaluation role=holdout,
# out of the same file. Measured on the built store: 110099 segments from 59 voices
# against 39053 from 21, with no voice in both.
#
# The Griffin-Lim floor (#89) is a separate arm and is not in this grid. There is no
# diffusion row: audio has no comparable diffusion work on this task to answer to, and
# the few-step contrast is already carried by the four flow-matching arms (see #87).
#
# Budget: 70.5 s/epoch measured by the #86 smoke run, so 2 epochs is ~2.5 min and about
# 0.04 GPU-h per task -- under an hour of GPU for the twenty, against ~7300 remaining.
# The wall clock below is many times the measured run because an epoch here is thirty
# times a knee epoch and the store is read from lustre if staging is skipped.
#SBATCH --job-name=cyfm-table6
#SBATCH --partition=lem-gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:hopper:1
#SBATCH --array=0-19
#SBATCH --output=/path/to/project/CyFM/logs/%x-%A_%a.txt
#SBATCH --error=/path/to/project/CyFM/logs/%x-%A_%a.err
set -u
source /path/to/project/CyFM/scripts/cluster/env.sh

PREFIX="${PREFIX:-t6_}"
GEOMS="${GEOMS:-cylindrical euclidean}"
COUPLINGS="${COUPLINGS:-independent ot}"
SEEDS="${SEEDS:-0 1 2 3 4}"
EPOCHS="${EPOCHS:-2}"
STORE_DIR="${STORE_DIR:-$CYFM_ROOT/data/librispeech_stft}"

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

for f in librispeech.h5 librispeech.json; do
  [ -f "$STORE_DIR/$f" ] || { echo "missing $STORE_DIR/$f; run build_stft_store.sbatch first"; exit 2; }
done

# GPU nodes carry 7 TB of node-local disk, unlike the CPU partition where TmpDisk=0.
# Two epochs over 110k segments is a lot of random reads for lustre to serve; ~5 GB
# copied once takes seconds.
STAGE="${TMPDIR:-/tmp/$USER}/table6-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE"
t0=$(date +%s)
cp "$STORE_DIR"/librispeech.h5 "$STORE_DIR"/librispeech.json "$STAGE/"
echo "staged $(du -sh "$STAGE" | cut -f1) in $(( $(date +%s) - t0 ))s to $STAGE"

echo "task=$SLURM_ARRAY_TASK_ID host=$(hostname) name=$NAME epochs=$EPOCHS"
EXPERIMENT=table6_audio \
DATA_DIR="$STAGE" \
HOLDOUT_ROLE=holdout \
  bash scripts/paper/run_arm.sh "$NAME" native "$M" "$C" "$SEED" "$EPOCHS"

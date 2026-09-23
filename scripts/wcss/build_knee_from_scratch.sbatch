#!/usr/bin/env bash
# Stage two of the knee intake: build the part stores on a GPU, reading the volumes the
# first-stage jobs already downloaded and extracted.
#
#   sbatch --array=0-4 scripts/wcss/build_knee_from_scratch.sbatch
#
# Why this exists. The first stage calibrated ESPIRiT on CPU, over the whole volume, at
# ~376 s each, which projected to 30-80 h per archive against a 48 h wall. On an H100,
# restricted to the 11 slices the store keeps, the same calibration measures 1.5 s --
# about 250x -- so the intake stops being bound by ESPIRiT and becomes bound by reading
# lustre.
#
# It reads the first-stage jobs' $TMPDIR directly rather than a copy. $TMPDIR here is a
# shared lustre mount, so another job can see it, and copying 875 GB only to read it
# back would now cost more than the computation does. Those jobs must therefore still
# be alive: their EXIT trap wipes the directory. They also delete each volume as they
# consume it, at roughly one per six minutes, which this outruns by two orders of
# magnitude -- but it is a race, so a volume that vanishes mid-run is skipped by the
# builder rather than being fatal.
#SBATCH --job-name=cyfm-knee-gpu
#SBATCH --partition=lem-gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:hopper:1
#SBATCH --array=0-4
#SBATCH --output=/path/to/project/CyFM/logs/%x-%A_%a.txt
#SBATCH --error=/path/to/project/CyFM/logs/%x-%A_%a.err
set -u
source /path/to/project/CyFM/scripts/wcss/env.sh

mapfile -t SRCS < <(ls -d /lustre/tmp/slurm/*/knee-*/extract/multicoil_train 2>/dev/null | sort)
if [ "${#SRCS[@]}" -eq 0 ]; then
  echo "no first-stage scratch left; are those jobs still running?"
  exit 2
fi
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#SRCS[@]}" ]; then
  echo "task $SLURM_ARRAY_TASK_ID out of range: found ${#SRCS[@]} source dir(s)"
  exit 2
fi
SRC="${SRCS[$SLURM_ARRAY_TASK_ID]}"
PART=$(echo "$SRC" | grep -o 'knee-[0-9]*-[0-9]*')
PARTS_DIR="$CYFM_ROOT/data/knee_pd/parts/train"
mkdir -p "$PARTS_DIR"

# Sidecars go to this job's own scratch and die with it: they are the same size as the
# k-space and are never read again once the slices are in the store.
WORK="${TMPDIR:-/tmp/$USER}/gpu-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/sens"

echo "task=$SLURM_ARRAY_TASK_ID host=$(hostname) src=$SRC part=$PART"
echo "volumes=$(find "$SRC" -maxdepth 1 -name '*.h5' | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

t0=$(date +%s)
uv run --no-sync python scripts/data/build_knee_pd_store.py \
  --raw-dir "$SRC" \
  --sens-dir "$WORK/sens" \
  --out "$PARTS_DIR/$PART.h5" \
  --split train \
  --device auto
code=$?
echo "build_exit=$code seconds=$(( $(date +%s) - t0 ))"
exit "$code"

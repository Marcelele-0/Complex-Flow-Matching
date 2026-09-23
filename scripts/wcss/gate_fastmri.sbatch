#!/usr/bin/env bash
# Issue #76, the fastMRI gate, as one GPU job: the batch size at 320x320, one timed
# epoch, an evaluation from held-out volumes, and the amplitude/phase sanity check.
#
#   sbatch scripts/wcss/gate_fastmri.sbatch
#   BATCH=16 sbatch scripts/wcss/gate_fastmri.sbatch    # time a chosen batch instead
#
# Needs data/knee_pd/val.h5 and val.json under the project root and nothing else: no
# raw k-space, no NYU links, no ESPIRiT. Everything it measures lands in the log and in
# outputs/gate/, ending with the lines to copy into the cohort's record.
#
# The probe runs both geometries, because every arm of Table 5 shares one batch size and
# the hungrier arm is the binding one. It runs each twice: on the whole H100, and capped
# at 15 GiB to stand in for the 16 GB RTX 4070 Ti SUPER the issue also asks about.
#SBATCH --job-name=cyfm-gate
#SBATCH --partition=lem-gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=/path/to/project/CyFM/logs/%x-%j.txt
#SBATCH --error=/path/to/project/CyFM/logs/%x-%j.err
set -u
source /path/to/project/CyFM/scripts/wcss/env.sh

EXPERIMENT=table5_fastmri
MANIFOLD="${MANIFOLD:-cylindrical}"
COUPLING="${COUPLING:-ot}"
STORE="$CYFM_ROOT/data/knee_pd/val.h5"
GATE_DIR="$CYFM_ROOT/outputs/gate/$SLURM_JOB_ID"
mkdir -p "$GATE_DIR"

if [ ! -f "$STORE" ] || [ ! -f "${STORE%.h5}.json" ]; then
  echo "no $STORE with its .json manifest. Copy the validation store there first (#78);"
  echo "training reads the store and never builds it."
  exit 2
fi

hostname
nvidia-smi --query-gpu=name,memory.total --format=csv

echo
echo "== 1. data sanity: amplitude and phase =="
uv run --no-sync python scripts/data/knee_store_panels.py \
  --store "$STORE" --out "$GATE_DIR/knee_panels.png" | tee "$GATE_DIR/panels.txt"

echo
echo "== 2. largest batch at 320x320 =="
for M in cylindrical euclidean; do
  for CAP in 0 15; do
    if [ "$CAP" = 0 ]; then TAG="${M}_h100"; else TAG="${M}_${CAP}gib"; fi
    echo "-- $TAG"
    uv run --no-sync python scripts/paper/probe_batch_size.py --side 320 --memory-gib "$CAP" \
      +experiment="$EXPERIMENT" manifold="$M" training.coupling="$COUPLING" \
      | tee "$GATE_DIR/probe_$TAG.txt"
  done
done

# The timed run uses the largest power of two that both geometries fit on the H100,
# unless BATCH says otherwise.
PROBED="$(sed -n 's/.*max_power_of_two=\([0-9]*\).*/\1/p' "$GATE_DIR"/probe_*_h100.txt | sort -n | head -1)"
BATCH="${BATCH:-$PROBED}"
if [ -z "$BATCH" ] || [ "$BATCH" = 0 ]; then
  echo "the probe found no batch that fits; see $GATE_DIR/probe_*.txt"
  exit 1
fi

echo
echo "== 3. one timed epoch, then evaluation from held-out volumes (batch $BATCH) =="
NAME="gate_${MANIFOLD}_${COUPLING}_b${BATCH}_j${SLURM_JOB_ID}"
EXPERIMENT="$EXPERIMENT" HOLDOUT_ROLE=holdout BATCH="$BATCH" \
  bash scripts/paper/run_arm.sh "$NAME" native "$MANIFOLD" "$COUPLING" 0 1 2>&1 \
  | tee "$GATE_DIR/epoch_and_eval.txt"

METRICS="$(find "$CYFM_ROOT/outputs/evaluate/${NAME}_eval" -name metrics.json 2>/dev/null | sort | tail -1)"

echo
echo "== numbers for the cohort record =="
grep -H "max_batch=" "$GATE_DIR"/probe_*.txt | sed "s#$GATE_DIR/##"
grep -h "epoch_seconds=" "$GATE_DIR/epoch_and_eval.txt"
grep -h "phase_step_median=\|VERDICT" "$GATE_DIR/panels.txt"
echo "metrics.json: ${METRICS:-MISSING -- evaluation did not finish, see $GATE_DIR/epoch_and_eval.txt}"
echo "panels:       $GATE_DIR/knee_panels.png"

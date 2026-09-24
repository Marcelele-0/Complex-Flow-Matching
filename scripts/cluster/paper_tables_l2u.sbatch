#!/usr/bin/env bash
# Tables 2 and 3 of the paper retrained with the fair loss (L2, unweighted phase),
# one array task per run:
#   {64, 32, 16} x {cylindrical, euclidean} x {independent, ot} x seeds {0, 1, 2}
# = 36 tasks, 64x64 first because those are the longest.
#
#   sbatch scripts/cluster/paper_tables_l2u.sbatch
#
# Run names carry the prefix l2u_ and otherwise match the table scripts, so the
# tables print with:
#   uv run python scripts/paper/paper_tables.py --prefix l2u_ --seeds 0 1 2
#
# The grid can be narrowed with environment variables; --array must then match
# the task count (|SIDES| x |GEOMS| x |COUPLINGS| x |SEEDS|). Examples:
#   seeds 3 and 4 of the same grid (24 tasks):
#     sbatch --array=0-23 --export=ALL,SEEDS="3 4" scripts/cluster/paper_tables_l2u.sbatch
#   Cartesian with the v1 loss at 64x64, 5 seeds (10 tasks):
#     sbatch --array=0-9 --export=ALL,SIDES=64,GEOMS=euclidean,LOSS_MODE=l1w,PREFIX=l1w_,SEEDS="0 1 2 3 4" \
#       scripts/cluster/paper_tables_l2u.sbatch
#   the cylinder with L1 and an unweighted phase term, all sizes, 5 seeds (30 tasks):
#     sbatch --array=0-29 --export=ALL,GEOMS=cylindrical,LOSS_MODE=l1u,PREFIX=l1u_,SEEDS="0 1 2 3 4" \
#       scripts/cluster/paper_tables_l2u.sbatch
#SBATCH --job-name=cyfm-l2u
#SBATCH --partition=lem-gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:hopper:1
#SBATCH --array=0-35
#SBATCH --output=/path/to/project/CyFM/logs/%x-%A_%a.txt
#SBATCH --error=/path/to/project/CyFM/logs/%x-%A_%a.err
set -u
source /path/to/project/CyFM/scripts/cluster/env.sh
PREFIX="${PREFIX:-l2u_}"
SIDES="${SIDES:-64 32 16}"
GEOMS="${GEOMS:-cylindrical euclidean}"
COUPLINGS="${COUPLINGS:-independent ot}"
SEEDS="${SEEDS:-0 1 2}"
export LOSS_MODE="${LOSS_MODE:-l2u}"

TASKS=()
for SIDE in $SIDES; do
  for M in $GEOMS; do
    for C in $COUPLINGS; do
      for SEED in $SEEDS; do
        TASKS+=("$SIDE $M $C $SEED")
      done
    done
  done
done

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#TASKS[@]}" ]; then
  echo "task $SLURM_ARRAY_TASK_ID out of range: grid has ${#TASKS[@]} tasks"
  exit 2
fi
read -r SIDE M C SEED <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
if [ "$SIDE" = 64 ]; then
  NAME="${PREFIX}un_${M}_${C}_scnull_s${SEED}"
else
  NAME="${PREFIX}unsz_${M}_${C}_${SIDE}_s${SEED}"
fi

echo "task=$SLURM_ARRAY_TASK_ID host=$(hostname) name=$NAME loss=$LOSS_MODE"
bash scripts/paper/run_arm.sh "$NAME" "$SIDE" "$M" "$C" "$SEED" 40

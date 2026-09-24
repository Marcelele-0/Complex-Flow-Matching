#!/usr/bin/env bash
# Table 7 (#81): re-evaluate the stored l2u checkpoints on a dense step grid, so the
# question can be turned around -- not "what error at k steps" but "how many steps to
# reach a given error".
#
#   sbatch scripts/cluster/dense_nfe_sweep.sbatch
#
# Evaluation only. No training: it reads the 60 runs already in outputs/train/ and
# writes new metrics under a _dense suffix, so the existing archives are untouched.
#
# Today nothing is measured between k=8 and k=100, which is exactly where the curves
# cross, so the sparse grid cannot answer the question. The dense grid adds 16, 24, 32,
# 48 and 64.
#SBATCH --job-name=cyfm-dense-nfe
#SBATCH --partition=lem-gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:hopper:1
#SBATCH --array=0-59
#SBATCH --output=/path/to/project/CyFM/logs/%x-%A_%a.txt
#SBATCH --error=/path/to/project/CyFM/logs/%x-%A_%a.err
set -u
source /path/to/project/CyFM/scripts/cluster/env.sh
cd "$CYFM_ROOT"

NFE="${NFE:-[1,2,4,8,16,24,32,48,64,100]}"

mapfile -t RUNS < <(ls -1d outputs/train/l2u_* 2>/dev/null | sed 's|.*/||' | sort)
if [ "${#RUNS[@]}" -eq 0 ]; then
  echo "no l2u_* runs under outputs/train"; exit 2
fi
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#RUNS[@]}" ]; then
  echo "task $SLURM_ARRAY_TASK_ID out of range: found ${#RUNS[@]} run(s)"; exit 2
fi
NAME="${RUNS[$SLURM_ARRAY_TASK_ID]}"

# l2u_un_<manifold>_<coupling>_scnull_s<seed>          -> 64x64
# l2u_unsz_<manifold>_<coupling>_<side>_s<seed>        -> 16 or 32
SEED="${NAME##*_s}"
if [[ "$NAME" == *_scnull_* ]]; then
  SIDE=native
  REST="${NAME#l2u_un_}"; REST="${REST%_scnull_s$SEED}"
else
  REST="${NAME#l2u_unsz_}"; REST="${REST%_s$SEED}"
  SIDE="${REST##*_}"; REST="${REST%_$SIDE}"
fi
M="${REST%%_*}"
C="${REST#*_}"

COMMON=(+experiment=paper_unet manifold="$M" training.coupling="$C")
if [ "$SIDE" != native ]; then
  COMMON+=(dataset.crop_size="[$SIDE,$SIDE]")
fi

echo "task=$SLURM_ARRAY_TASK_ID name=$NAME manifold=$M coupling=$C side=$SIDE seed=$SEED"
t0=$(date +%s)
uv run --no-sync python -m cyfm.evaluate "${COMMON[@]}" \
  evaluate.run_name="$NAME" evaluate.num_fields=64 evaluate.seed="$SEED" \
  evaluate.nfe="$NFE" logging.experiment_name="${NAME}_dense" logging.use_wandb=false
code=$?
echo "eval_exit=$code seconds=$(( $(date +%s) - t0 ))"
exit "$code"

#!/usr/bin/env bash
# One-time environment build for CyFM on HPC cluster, plus a timing probe.
#
#   sbatch scripts/wcss/setup_env.sbatch
#
# Builds the uv venv on a compute node (the login node caps users at 1 GB RAM),
# checks that torch sees the H100, then trains one 64x64 arm for 2 epochs and
# evaluates it, printing wall-clock seconds so the 40-epoch array can be budgeted.
#SBATCH --job-name=cyfm-setup
#SBATCH --partition=lem-gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32GB
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=/path/to/project/CyFM/logs/%x-%j.txt
#SBATCH --error=/path/to/project/CyFM/logs/%x-%j.err
set -u
source /path/to/project/CyFM/scripts/wcss/env.sh

hostname
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
curl -sS -o /dev/null -w "internet: %{http_code}\n" --max-time 10 https://pypi.org/simple/ \
  || echo "internet: none"

t0=$(date +%s)
uv sync --frozen --no-dev
code=$?
echo "uv_sync_exit=$code seconds=$(( $(date +%s) - t0 ))"
if [ "$code" -ne 0 ]; then exit "$code"; fi

uv run --no-sync python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

t1=$(date +%s)
bash scripts/paper/run_arm.sh timing_un_cylindrical_ot_64 64 cylindrical ot 0 2
code=$?
echo "timing_exit=$code seconds_2_epochs_plus_eval=$(( $(date +%s) - t1 ))"
exit "$code"

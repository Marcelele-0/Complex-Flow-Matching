#!/bin/bash
#SBATCH --job-name=cfm-euc-train
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_euc_%j.out
#SBATCH --error=logs/train_euc_%j.err

set -euo pipefail

# Ensure working directory is repo root
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
fi

mkdir -p logs

echo "========================================================"
echo "Job ID:    ${SLURM_JOB_ID:-N/A}"
echo "Node:      $(hostname)"
echo "Start:     $(date)"
echo "Directory: $(pwd)"
echo "GPU:       $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo 'No GPU info')"
echo "========================================================"

# Environment & PATH setup
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Load CUDA modules on PLGrid if available
if command -v module &> /dev/null; then
    module purge 2>/dev/null || true
    module load CUDA/12.4 2>/dev/null || module load CUDA/12.2 2>/dev/null || module load CUDA/12.1 2>/dev/null || module load CUDA 2>/dev/null || true
fi

# Run Euclidean Flow Matching baseline training
uv run python -m cfm.train \
    manifold=euclidean \
    logging.experiment_name=euclidean_final \
    dataset.data_dir=data/skm-tea/v1-release \
    training.epochs=100 \
    training.seed=0 \
    training.compile=true \
    training.matmul_precision=high \
    "$@"

echo "========================================================"
echo "Training finished at: $(date)"
echo "========================================================"

#!/bin/bash
#SBATCH --job-name=cfm-eval-both
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_both_%j.out
#SBATCH --error=logs/eval_both_%j.err

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

echo ""
echo "========================================================"
echo "Step 1/2: Evaluating Cylindrical Model"
echo "========================================================"
uv run python -m cfm.evaluate \
    manifold=cylindrical \
    evaluate.run_name=cylindrical_final \
    evaluate.split=test \
    dataset.data_dir=data/skm-tea/v1-release \
    "$@"

echo ""
echo "========================================================"
echo "Step 2/2: Evaluating Euclidean Model"
echo "========================================================"
uv run python -m cfm.evaluate \
    manifold=euclidean \
    evaluate.run_name=euclidean_final \
    evaluate.split=test \
    dataset.data_dir=data/skm-tea/v1-release \
    "$@"

echo ""
echo "========================================================"
echo "Evaluation completed at: $(date)"
echo "Metrics summary files:"
echo "  - outputs/evaluate/cylindrical_final/*/metrics.json"
echo "  - outputs/evaluate/euclidean_final/*/metrics.json"
echo "========================================================"

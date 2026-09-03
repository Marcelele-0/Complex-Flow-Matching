#!/bin/bash
#SBATCH --job-name=cfm-download-skm-tea
#SBATCH --partition=plgrid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err

set -euo pipefail

# Ensure working directory is repo root
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
fi

mkdir -p logs data/skm-tea/v1-release

echo "========================================================"
echo "Job ID:    ${SLURM_JOB_ID:-N/A}"
echo "Node:      $(hostname)"
echo "Start:     $(date)"
echo "Directory: $(pwd)"
echo "========================================================"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Check for AWS CLI availability
if ! command -v aws &> /dev/null; then
    echo "AWS CLI not found in PATH. Checking modules or attempting local install..."
    if command -v module &> /dev/null; then
        module load awscli 2>/dev/null || module load aws-cli 2>/dev/null || true
    fi
    if ! command -v aws &> /dev/null; then
        echo "Installing awscli via uv tool or pip..."
        uv tool install awscli 2>/dev/null || pip install --user awscli
    fi
fi

echo "Starting download of SKM-TEA dataset from S3..."
echo "Source:      s3://skm-tea/v1-release/"
echo "Destination: data/skm-tea/v1-release/"

if command -v aws &> /dev/null; then
    aws s3 cp --no-sign-request --recursive s3://skm-tea/v1-release/ data/skm-tea/v1-release/
else
    uv tool run awscli s3 cp --no-sign-request --recursive s3://skm-tea/v1-release/ data/skm-tea/v1-release/
fi

echo "========================================================"
echo "Download completed at: $(date)"
echo "Downloaded directory size: $(du -sh data/skm-tea/v1-release/ 2>/dev/null || echo 'N/A')"
echo "========================================================"

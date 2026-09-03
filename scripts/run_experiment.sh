#!/bin/bash
# Master pipeline script to submit SKM-TEA download, training, and evaluation on PLGrid SLURM cluster.
#
# Usage:
#   ./scripts/run_experiment.sh                  # Full pipeline (download -> train both -> eval)
#   ./scripts/run_experiment.sh --skip-download  # Skip download step
#   ./scripts/run_experiment.sh --eval-only      # Run evaluation only on existing checkpoints

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs outputs data/skm-tea/v1-release

SKIP_DOWNLOAD=false
EVAL_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --skip-download)
            SKIP_DOWNLOAD=true
            ;;
        --eval-only)
            EVAL_ONLY=true
            ;;
        --help|-h)
            echo "Usage: $0 [--skip-download] [--eval-only]"
            echo ""
            echo "Options:"
            echo "  --skip-download   Skip dataset download from S3 and proceed directly to training"
            echo "  --eval-only       Submit evaluation job only for existing checkpoints"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Error: Unknown argument: $arg"
            echo "Usage: $0 [--skip-download] [--eval-only]"
            exit 1
            ;;
    esac
done

echo "=========================================================="
echo "CFM Experiment Pipeline Orchestrator (PLGrid / SLURM)"
echo "Root Directory: ${ROOT_DIR}"
echo "Start Time:     $(date)"
echo "=========================================================="

if [ "$EVAL_ONLY" = true ]; then
    echo "[1/1] Submitting evaluation job..."
    EVAL_JOB_OUT=$(sbatch --parsable scripts/slurm/evaluate_both.sbatch)
    EVAL_JOB_ID=$(echo "${EVAL_JOB_OUT}" | cut -d';' -f1)
    echo "  -> Submitted Evaluation Job ID: ${EVAL_JOB_ID}"
    echo ""
    echo "Monitor with: squeue -j ${EVAL_JOB_ID} or tail -f logs/eval_both_${EVAL_JOB_ID}.out"
    exit 0
fi

# Step 1: Dataset Download
DOWNLOAD_DEP=""
if [ "$SKIP_DOWNLOAD" = false ]; then
    if [ -d "data/skm-tea/v1-release/annotations" ] && [ -d "data/skm-tea/v1-release/records" ]; then
        echo "[1/3] Existing dataset detected in data/skm-tea/v1-release/. Skipping download job."
    else
        echo "[1/3] Submitting dataset download job..."
        DOWNLOAD_JOB_OUT=$(sbatch --parsable scripts/slurm/download_skm_tea.sbatch)
        DOWNLOAD_JOB_ID=$(echo "${DOWNLOAD_JOB_OUT}" | cut -d';' -f1)
        echo "  -> Submitted Download Job ID: ${DOWNLOAD_JOB_ID}"
        DOWNLOAD_DEP="--dependency=afterok:${DOWNLOAD_JOB_ID}"
    fi
else
    echo "[1/3] Skipping dataset download (--skip-download requested)."
fi

# Step 2: Submit Training Jobs
echo "[2/3] Submitting training jobs..."
if [ -n "$DOWNLOAD_DEP" ]; then
    CYL_JOB_OUT=$(sbatch --parsable ${DOWNLOAD_DEP} scripts/slurm/train_cylindrical.sbatch)
    EUC_JOB_OUT=$(sbatch --parsable ${DOWNLOAD_DEP} scripts/slurm/train_euclidean.sbatch)
else
    CYL_JOB_OUT=$(sbatch --parsable scripts/slurm/train_cylindrical.sbatch)
    EUC_JOB_OUT=$(sbatch --parsable scripts/slurm/train_euclidean.sbatch)
fi

CYL_JOB_ID=$(echo "${CYL_JOB_OUT}" | cut -d';' -f1)
EUC_JOB_ID=$(echo "${EUC_JOB_OUT}" | cut -d';' -f1)

echo "  -> Submitted Cylindrical Training Job ID: ${CYL_JOB_ID}"
echo "  -> Submitted Euclidean Training Job ID:   ${EUC_JOB_ID}"

# Step 3: Submit Evaluation Job (depends on both training runs completing successfully)
echo "[3/3] Submitting evaluation job (dependency: afterok:${CYL_JOB_ID}:${EUC_JOB_ID})..."
EVAL_JOB_OUT=$(sbatch --parsable --dependency=afterok:${CYL_JOB_ID}:${EUC_JOB_ID} scripts/slurm/evaluate_both.sbatch)
EVAL_JOB_ID=$(echo "${EVAL_JOB_OUT}" | cut -d';' -f1)
echo "  -> Submitted Evaluation Job ID:           ${EVAL_JOB_ID}"

echo ""
echo "=========================================================="
echo "Pipeline submitted successfully!"
echo "DAG Execution Structure:"
if [ -n "$DOWNLOAD_DEP" ]; then
    echo "  Download Job [${DOWNLOAD_JOB_ID}]"
    echo "      ├──> Train Cylindrical [${CYL_JOB_ID}] ──┐"
    echo "      └──> Train Euclidean   [${EUC_JOB_ID}] ──┴──> Evaluate Both [${EVAL_JOB_ID}]"
else
    echo "  Train Cylindrical [${CYL_JOB_ID}] ──┐"
    echo "  Train Euclidean   [${EUC_JOB_ID}] ──┴──> Evaluate Both [${EVAL_JOB_ID}]"
fi
echo "=========================================================="
echo "Useful Commands for Monitoring:"
echo "  Check queue status:  squeue -u \$USER"
echo "  Inspect job details: scontrol show job <JOB_ID>"
echo "  Follow logs:"
echo "    tail -f logs/train_cyl_${CYL_JOB_ID}.out"
echo "    tail -f logs/train_euc_${EUC_JOB_ID}.out"
echo "    tail -f logs/eval_both_${EVAL_JOB_ID}.out"
echo "=========================================================="

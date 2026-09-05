#!/bin/bash -l
# Multi-GPU CFM training on PLGrid A100 nodes.
#
# Submit through scripts/launch_slurm.sh, which resolves the account, the resource
# request and the storage group and dry-runs the job shape first. The directives
# below are the defaults for a direct `sbatch scripts/slurm/train_ddp.sbatch`; the
# launcher overrides them on the command line, which wins over any #SBATCH line.
#
# Every argument after the script name is forwarded verbatim to train.py as a Hydra
# override:
#
#   sbatch scripts/slurm/train_ddp.sbatch manifold=euclidean training.epochs=200
#
#SBATCH --job-name=cfm-train-ddp
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_ddp_%j.out
#SBATCH --error=logs/train_ddp_%j.err
# Requeue instead of dying when the node fails or the job is preempted, and append
# to the same log so a run that took three scheduler slots reads as one file.
#SBATCH --requeue
#SBATCH --open-mode=append
# Wake the batch script (B:) 600s before the wall clock, which is the window the run
# has to finish its epoch, flush last.pt and be put back in the queue. A job that
# simply hits its limit ends as TIMEOUT and is NOT requeued.
#SBATCH --signal=B:USR1@600

set -euo pipefail

# One task per node; torchrun forks the per-GPU workers underneath it.
GPUS_PER_NODE="${CFM_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"
PROJECT_NAME="${CFM_PROJECT:-complex-flow-matching}"
# Seconds to let training stop cleanly before requeueing regardless. Must stay under
# the --signal lead time above or the job is killed before it is put back.
PREEMPT_GRACE="${CFM_PREEMPT_GRACE:-480}"

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
fi
mkdir -p logs

# --- Heavy storage -----------------------------------------------------------
# Home is a ~10 GiB quota on Athena. The environment, the caches, the datasets and
# the outputs all live in group storage; only code stays in $HOME.
: "${PLG_GROUP:?Set PLG_GROUP to the writable group shown by hpc-fs}"
: "${PLG_GROUPS_STORAGE:?PLG_GROUPS_STORAGE is not defined on this host}"
HEAVY="$PLG_GROUPS_STORAGE/$PLG_GROUP/users/$USER/$PROJECT_NAME"
test -d "$HEAVY" || {
    echo "ERROR: $HEAVY does not exist. Run scripts/slurm/bootstrap_plgrid_storage.sh first." >&2
    exit 1
}

export XDG_CACHE_HOME="$HEAVY/cache"
export UV_CACHE_DIR="$HEAVY/cache/uv"
export PIP_CACHE_DIR="$HEAVY/cache/pip"
export TORCH_HOME="$HEAVY/cache/torch"
export TRITON_CACHE_DIR="$HEAVY/cache/triton"
export HF_HUB_CACHE="$HEAVY/cache/huggingface-hub"
export WANDB_DIR="$HEAVY/outputs"
# uv would otherwise build .venv/ in $HOME and blow the quota on the torch wheels.
export UV_PROJECT_ENVIRONMENT="$HEAVY/envs/a100"

echo "========================================================"
echo "Job ID:     ${SLURM_JOB_ID:-N/A}  (restart #${SLURM_RESTART_COUNT:-0})"
echo "Nodes:      ${SLURM_JOB_NUM_NODES:-1} x ${GPUS_PER_NODE} GPU"
echo "Node list:  ${SLURM_JOB_NODELIST:-$(hostname)}"
echo "Start:      $(date -Is)"
echo "Directory:  $(pwd)"
echo "Commit:     $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"
echo "Heavy:      $HEAVY"
echo "========================================================"

# --- Modules and PATH --------------------------------------------------------
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export PYTHONUNBUFFERED=1
# cpus-per-task is the whole node's share for the single srun task, so each of the
# GPUS_PER_NODE workers gets its slice rather than all of them spawning a full set
# of OpenMP threads and thrashing.
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-32} / GPUS_PER_NODE ))

# The published PyTorch module on Athena is far older than this project's
# torch>=2.11, so the CUDA runtime is all we take from the module system and the
# environment itself comes from uv. Override the list with CFM_MODULES if the
# catalog moves; verify with `module spider CUDA` on the login node first.
if command -v module &> /dev/null; then
    module purge 2>/dev/null || true
    if [[ -n "${CFM_MODULES:-}" ]]; then
        # shellcheck disable=SC2086
        module load ${CFM_MODULES}
    else
        module load CUDA/12.4 2>/dev/null \
            || module load CUDA/12.2 2>/dev/null \
            || module load CUDA/12.1 2>/dev/null \
            || module load CUDA 2>/dev/null \
            || true
    fi
    module list 2>&1 || true
fi

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# --- Rendezvous --------------------------------------------------------------
# torchrun needs one address every node agrees on. The port is derived from the job
# ID so two jobs sharing a node cannot collide on it.
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}" | head -n1)"
MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
export MASTER_ADDR MASTER_PORT
echo "Rendezvous: $MASTER_ADDR:$MASTER_PORT"

# --- Preemption --------------------------------------------------------------
# A flag file rather than a relayed signal. Slurm signals this batch script, but the
# training processes are three launchers away (srun -> uv -> torchrun agent -> N
# workers) and the agent has no SIGUSR1 handler, so a forwarded signal kills the
# workers outright instead of letting them finish the epoch and save. The file is
# visible to every rank on every node and no launcher can swallow it.
PREEMPT_FILE="$HEAVY/outputs/preempt-${SLURM_JOB_ID}"
rm -f "$PREEMPT_FILE"

requeue_now() {
    echo "[slurm] $(date -Is) requeueing job ${SLURM_JOB_ID}"
    scontrol requeue "$SLURM_JOB_ID" || echo "[slurm] requeue refused; resubmit by hand" >&2
}

on_grace_signal() {
    echo "[slurm] $(date -Is) USR1: wall clock is close, asking training to stop."
    touch "$PREEMPT_FILE"
    # If training does not reach an epoch boundary inside the grace window, requeue
    # anyway: a job that reaches its wall clock ends as TIMEOUT, and TIMEOUT is not
    # requeued. last.pt is then one epoch stale rather than current.
    ( sleep "$PREEMPT_GRACE"; requeue_now ) &
    WATCHDOG_PID=$!
    requeue_requested=1
}

requeue_requested=0
WATCHDOG_PID=""
trap on_grace_signal USR1

# --- Train -------------------------------------------------------------------
# One srun task per node, GPUS_PER_NODE torchrun workers inside it. auto_resume is
# forced on here and only here: on a cluster a restart is the scheduler's decision,
# so the run has to come back where it stopped rather than at epoch 0.
srun --kill-on-bad-exit=1 uv run torchrun \
    --nnodes="${SLURM_JOB_NUM_NODES:-1}" \
    --nproc-per-node="${GPUS_PER_NODE}" \
    --node-rank="${SLURM_NODEID:-0}" \
    --rdzv-id="${SLURM_JOB_ID}" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    src/cfm/train.py \
    training.auto_resume=true \
    "training.preempt_file=$PREEMPT_FILE" \
    "$@" &
TRAIN_PID=$!

status=0
wait "$TRAIN_PID" || status=$?
# A trap interrupts wait and makes it return >128 without reaping the child; the
# second wait collects the real exit status.
if (( status > 128 )); then
    status=0
    wait "$TRAIN_PID" || status=$?
fi

# The watchdog has usually already fired or is no longer needed; failing to kill it
# must not abort the requeue below, which is the whole point of the grace window.
[[ -n "$WATCHDOG_PID" ]] && kill "$WATCHDOG_PID" 2>/dev/null || true

echo "========================================================"
echo "Training process exited with status ${status} at $(date -Is)"

if (( requeue_requested )); then
    # Grace signal was seen, so this instance stopped short of the epoch budget.
    # A run that happened to finish inside the window comes back, sees
    # epochs_completed == training.epochs and exits in seconds.
    requeue_now
    exit 0
fi

echo "Training finished at: $(date -Is)"
echo "========================================================"
exit "$status"

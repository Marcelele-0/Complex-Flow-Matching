#!/usr/bin/env bash
# Submit multi-GPU CFM training to PLGrid (Athena, plgrid-gpu-a100).
#
# Dry-runs by default: without --submit this only asks Slurm to validate the job
# shape, which consumes no allocation. Check the printed request, then re-run with
# --submit.
#
#   ./scripts/launch_slurm.sh --gpus 4 -- manifold=cylindrical
#   ./scripts/launch_slurm.sh --gpus 4 --submit -- manifold=euclidean
#   ./scripts/launch_slurm.sh --nodes 2 --gpus 4 --time 48:00:00 --submit
#
# Everything after `--` is forwarded to train.py as Hydra overrides.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB_SCRIPT="scripts/slurm/train_ddp.sbatch"

# --- Defaults ----------------------------------------------------------------
# Verified against PLGrid on 2026-07-22 by ofurman/plgrid-skill. Grants, partitions
# and modules change: confirm with hpc-grants / hpc-fs / sinfo before a first run.
PARTITION="${PLG_PARTITION:-plgrid-gpu-a100}"
ACCOUNT="${PLG_GRANT:-}"
NODES=1
GPUS=4
CPUS_PER_GPU=8
MEM_PER_GPU_GB=60
TIME="24:00:00"
JOB_NAME=""
EXPERIMENT=""
SUBMIT=0
RESUME=1
QOS=""

usage() {
    sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  -g, --gpus N            GPUs per node (default 4; an Athena A100 node has 8)
  -N, --nodes N           Nodes (default 1)
  -t, --time HH:MM:SS     Wall clock (default 24:00:00)
  -c, --cpus-per-gpu N    CPU cores per GPU (default 8)
      --mem-per-gpu GB    Host memory per GPU in GiB (default 60)
  -A, --account NAME      Slurm account (default $PLG_GRANT from .env)
  -p, --partition NAME    Partition (default $PLG_PARTITION, else plgrid-gpu-a100)
      --qos NAME          Slurm QOS, if your grant needs one
  -J, --job-name NAME     Job name (default cfm-<experiment>)
  -e, --experiment NAME   logging.experiment_name; also names the resume state
      --no-resume         Start from epoch 0 even if a resume state exists
  -s, --submit            Actually submit (default is sbatch --test-only)
  -h, --help              This message
  --                      Everything after is passed to train.py as Hydra overrides
EOF
}

# --- Arguments ---------------------------------------------------------------
OVERRIDES=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -g|--gpus)          GPUS="$2"; shift 2 ;;
        -N|--nodes)         NODES="$2"; shift 2 ;;
        -t|--time)          TIME="$2"; shift 2 ;;
        -c|--cpus-per-gpu)  CPUS_PER_GPU="$2"; shift 2 ;;
        --mem-per-gpu)      MEM_PER_GPU_GB="$2"; shift 2 ;;
        -A|--account)       ACCOUNT="$2"; shift 2 ;;
        -p|--partition)     PARTITION="$2"; shift 2 ;;
        --qos)              QOS="$2"; shift 2 ;;
        -J|--job-name)      JOB_NAME="$2"; shift 2 ;;
        -e|--experiment)    EXPERIMENT="$2"; shift 2 ;;
        --no-resume)        RESUME=0; shift ;;
        -s|--submit)        SUBMIT=1; shift ;;
        -h|--help)          usage; exit 0 ;;
        --)                 shift; OVERRIDES+=("$@"); break ;;
        *)                  echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# --- Non-secret settings from .env -------------------------------------------
# Read key by key rather than sourced: an env file is not trusted input, and
# sourcing one would run whatever it contains.
if [[ -f "$REPO_ROOT/.env" ]]; then
    while IFS='=' read -r key value; do
        case "$key" in
            PLG_GROUP|PLG_GRANT|PLG_PARTITION)
                [[ -n "${!key:-}" ]] || export "$key=$value" ;;
        esac
    done < <(grep -E '^(PLG_GROUP|PLG_GRANT|PLG_PARTITION)=' "$REPO_ROOT/.env" || true)
    ACCOUNT="${ACCOUNT:-${PLG_GRANT:-}}"
    [[ "$PARTITION" == "plgrid-gpu-a100" && -n "${PLG_PARTITION:-}" ]] && PARTITION="$PLG_PARTITION"
fi

cd "$REPO_ROOT"
test -f "$JOB_SCRIPT" || { echo "Missing $JOB_SCRIPT" >&2; exit 1; }
mkdir -p logs

# --- Preflight ---------------------------------------------------------------
if ! command -v sbatch &> /dev/null; then
    echo "sbatch not found: run this on a PLGrid login node, not your laptop." >&2
    exit 1
fi

if [[ -z "$ACCOUNT" ]]; then
    echo "No Slurm account. Pass --account, or set PLG_GRANT in .env." >&2
    echo "List your grants with: hpc-grants" >&2
    exit 1
fi

if [[ -z "${PLG_GROUP:-}" ]]; then
    echo "PLG_GROUP is unset. The job writes its environment, caches and outputs to" >&2
    echo "group storage, so it needs the writable group hpc-fs reports:" >&2
    echo "  hpc-fs && export PLG_GROUP=<group>" >&2
    exit 1
fi

# Live check beats the recorded default; partitions are renamed and retired.
if ! sinfo -h -p "$PARTITION" -o '%P' | grep -q .; then
    echo "Partition '$PARTITION' is not visible here. Available GPU partitions:" >&2
    sinfo -h -o '%P' | sort -u >&2
    exit 1
fi

# --- Derived request ---------------------------------------------------------
CPUS_PER_TASK=$(( GPUS * CPUS_PER_GPU ))
MEM_GB=$(( GPUS * MEM_PER_GPU_GB ))
WORLD_SIZE=$(( NODES * GPUS ))
[[ -n "$JOB_NAME" ]] || JOB_NAME="cfm-${EXPERIMENT:-train}-${WORLD_SIZE}gpu"

# Hydra rejects the same key twice, so a flag only contributes an override when the
# caller has not already written one after `--`.
has_override() {
    local key="$1"
    for value in ${OVERRIDES[@]+"${OVERRIDES[@]}"}; do
        [[ "$value" == "$key="* ]] && return 0
    done
    return 1
}

TRAIN_ARGS=()
[[ -n "$EXPERIMENT" ]] && ! has_override logging.experiment_name \
    && TRAIN_ARGS+=("logging.experiment_name=$EXPERIMENT")
# One core is left for the main process to feed the GPU.
has_override training.num_workers \
    || TRAIN_ARGS+=("training.num_workers=$(( CPUS_PER_GPU > 1 ? CPUS_PER_GPU - 1 : 0 ))")
# train_ddp.sbatch turns auto_resume on; this is the only way back off it.
(( RESUME )) || TRAIN_ARGS+=("training.auto_resume=false")
TRAIN_ARGS+=(${OVERRIDES[@]+"${OVERRIDES[@]}"})

SBATCH_ARGS=(
    --job-name="$JOB_NAME"
    --account="$ACCOUNT"
    --partition="$PARTITION"
    --nodes="$NODES"
    --ntasks-per-node=1
    --gres="gpu:$GPUS"
    --cpus-per-task="$CPUS_PER_TASK"
    --mem="${MEM_GB}G"
    --time="$TIME"
    --output="logs/${JOB_NAME}_%j.out"
    --error="logs/${JOB_NAME}_%j.err"
    --export="ALL,CFM_GPUS_PER_NODE=$GPUS"
)
[[ -n "$QOS" ]] && SBATCH_ARGS+=(--qos="$QOS")

cat <<EOF
========================================================
  Partition:   $PARTITION
  Account:     $ACCOUNT
  Job name:    $JOB_NAME
  Layout:      $NODES node(s) x $GPUS GPU = $WORLD_SIZE ranks
  Per node:    $CPUS_PER_TASK CPU, ${MEM_GB}G RAM
  Wall clock:  $TIME
  Storage:     \$PLG_GROUPS_STORAGE/$PLG_GROUP/users/$USER/${CFM_PROJECT:-complex-flow-matching}
  Overrides:   ${TRAIN_ARGS[*]:-<none>}
========================================================
EOF

# --- Validate, then submit ---------------------------------------------------
# --test-only never consumes allocation, so it runs on every path including the
# real one: a request Slurm cannot satisfy should fail here, not after a queue wait.
echo "Validating job shape (sbatch --test-only)..."
sbatch --test-only "${SBATCH_ARGS[@]}" "$JOB_SCRIPT" ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}

if (( ! SUBMIT )); then
    echo
    echo "Dry run only. Re-run with --submit to consume allocation."
    exit 0
fi

JOB_ID="$(sbatch --parsable "${SBATCH_ARGS[@]}" "$JOB_SCRIPT" ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"})"
printf '%s\n' "$JOB_ID" > "logs/${JOB_NAME}.jobid"

echo
echo "Submitted job $JOB_ID (id recorded in logs/${JOB_NAME}.jobid)"
echo "  squeue -j $JOB_ID"
echo "  tail -f logs/${JOB_NAME}_${JOB_ID}.out"
echo "  sacct -j $JOB_ID --format=JobID,JobName,State,Elapsed,AllocTRES,ExitCode"
echo
echo "A submitted job ID is not a result: verify with sacct and the logs before"
echo "reporting the run as done."

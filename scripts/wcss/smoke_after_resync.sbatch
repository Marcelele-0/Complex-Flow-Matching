#!/usr/bin/env bash
# Verify a freshly resynced CyFM tree before spending GPU hours on a cohort.
#
#   sbatch scripts/wcss/smoke_after_resync.sbatch
#
# The cluster tree is an rsync copy rather than a checkout, so nothing there
# guarantees that what landed is what the repository holds: a stale package can sit
# beside the new one, the venv still points at the path the project had when it was
# installed, and an optional dependency moved behind an extra fails only at the call
# site. This runs the cheapest check for each of those, in the order that makes a
# failure interpretable -- environment, then imports, then the suite, then one real
# GPU step -- so the first thing that breaks names the cause.
#
# It is not a correctness check on the science. It answers "is this tree runnable",
# which is the question that has to be answered before a cohort is worth launching.
#SBATCH --job-name=cyfm-smoke
#SBATCH --partition=lem-gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32GB
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/CyFM/logs/%x-%j.txt
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/CyFM/logs/%x-%j.err
set -u
source /lustre/pd03/hpc-danbor2008-1756464546/CyFM/scripts/wcss/env.sh

echo "=== host and GPU"
hostname
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "tree revision: $(cat /lustre/pd03/hpc-danbor2008-1756464546/CyFM/GIT_REVISION 2>/dev/null)"

# The venv records the project as a path dependency, so a rename or a move leaves it
# importing something that is no longer there. Reinstall before trusting any import.
echo "=== uv sync"
t0=$(date +%s)
uv sync --frozen
echo "uv_sync_exit=$? seconds=$(( $(date +%s) - t0 ))"

# A stale package next to the new one is the failure this resync existed to prevent,
# and it is silent: both import, and the wrong one may win.
echo "=== package layout"
ls /lustre/pd03/hpc-danbor2008-1756464546/CyFM/src/
uv run --no-sync python -c "import cyfm, pathlib; print('cyfm at', pathlib.Path(cyfm.__file__).parent)"
uv run --no-sync python -c "import importlib.util; print('stale cfm importable:', importlib.util.find_spec('cfm') is not None)"

echo "=== test suite"
uv run --no-sync pytest tests/ -q

echo "=== entry point"
uv run --no-sync python -m cyfm.train --help 2>&1 | head -2

# Network-free end to end: dataset registry, manifold, prior and metrics. The value is
# platform-dependent in its last digits, so what matters is the band, not the digits.
echo "=== prior control (expect ~0.185)"
uv run --no-sync python scripts/prior_control.py --side 16 --num-fields 16

# The only part the checks above do not touch: the training path on a real GPU --
# bridge, coupling, loss, optimiser, checkpointing, then evaluation of what it wrote.
echo "=== two-epoch arm, 64x64 synthetic"
t0=$(date +%s)
LOSS_MODE=l2u bash scripts/paper/run_arm.sh smoke_resync 64 cylindrical ot 0 2
echo "arm_exit=$? seconds=$(( $(date +%s) - t0 ))"

echo "=== done"

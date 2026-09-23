# Environment for CyFM jobs on WCSS (ui.wcss.pl). Source it from a batch script.
#
# Layout, chosen so the whole group can use and inspect a run:
#   $PDDIR/CyFM            code, outputs, logs and caches, group-writable (setgid)
#   $HOME/.venv-cyfm       the uv environment, one per user, on NFS
#
# The venv stays out of lustre on purpose: it is tens of thousands of small files,
# which lustre serves badly, and it must not be shared between users. `uv` is told
# where it lives through UV_PROJECT_ENVIRONMENT, so `uv run --no-sync` works with
# the project itself on lustre.
#
# Paths are absolute on purpose: a bare `cd` on this site lands in the shared grant
# storage, not in $HOME.
CYFM_ROOT=/path/to/project/CyFM

# New files stay group-writable, so a teammate can rerun or clean up after us.
umask 0002

export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$HOME/.venv-cyfm"
export UV_CACHE_DIR="$CYFM_ROOT/cache/uv"
# Cache on lustre, venv on NFS: hard links cannot cross filesystems.
export UV_LINK_MODE=copy
export PYTHONNOUSERSITE=1
export WANDB_MODE=disabled
export TORCH_HOME="/tmp/$USER-torch"
export TRITON_CACHE_DIR="/tmp/$USER-triton"
export CUDA_CACHE_PATH="/tmp/$USER-cuda"
export TORCHINDUCTOR_CACHE_DIR="/tmp/$USER-inductor"
export MPLCONFIGDIR="/tmp/$USER-matplotlib"

cd "$CYFM_ROOT" || exit 1

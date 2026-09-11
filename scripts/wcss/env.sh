# Environment for CyFM jobs on WCSS (ui.wcss.pl). Source it from a batch script.
#
# Layout: code and the uv venv live in home (NFS, fast for many small files);
# outputs/ in the checkout is a symlink to the grant storage, and every cache is
# kept out of home. Paths are absolute on purpose: a bare `cd` on this site lands
# in the shared grant storage, not in $HOME.
CYFM_HOME=/home/marcelele/CyFM
CYFM_STORE=/lustre/pd03/hpc-danbor2008-1756464546/marcelele/CyFM

export PATH="/home/marcelele/.local/bin:$PATH"
export UV_CACHE_DIR="$CYFM_STORE/cache/uv"
# Cache on lustre, venv on NFS: hard links cannot cross filesystems.
export UV_LINK_MODE=copy
export PYTHONNOUSERSITE=1
export WANDB_MODE=disabled
export TORCH_HOME="/tmp/$USER-torch"
export TRITON_CACHE_DIR="/tmp/$USER-triton"
export CUDA_CACHE_PATH="/tmp/$USER-cuda"
export TORCHINDUCTOR_CACHE_DIR="/tmp/$USER-inductor"
export MPLCONFIGDIR="/tmp/$USER-matplotlib"

cd "$CYFM_HOME" || exit 1

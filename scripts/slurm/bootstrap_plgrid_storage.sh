#!/usr/bin/env bash
# Create this project's group-storage tree on a PLGrid cluster and link it into the
# code directory. Run once per cluster, on the login node, before the first job.
#
#   export PLG_GROUP="$(hpc-fs | ...)"   # the writable group hpc-fs reports
#   ./scripts/slurm/bootstrap_plgrid_storage.sh
#
# Home on Athena is a ~10 GiB quota, which a single torch environment fills. Code
# and Slurm files stay in $HOME so the login and compute nodes both see them; the
# environment, caches, datasets, checkpoints and outputs go to group storage and are
# reached through project-scoped symlinks.
#
# Adapted from ofurman/plgrid-skill (plgrid-run, commit 7f7fdf1). It refuses to
# replace anything that already exists, so it is safe to re-run.
set -euo pipefail

: "${PLG_GROUP:?Set PLG_GROUP to the writable group shown by hpc-fs}"
: "${PLG_GROUPS_STORAGE:?PLG_GROUPS_STORAGE is not defined; are you on a PLGrid host?}"

PROJECT_NAME="${CFM_PROJECT:-complex-flow-matching}"
# x86_64 on Athena A100. Read from the compute-node profile, not from uname here:
# some PLGrid clusters have login and compute nodes on different architectures.
COMPUTE_ARCH="${CFM_COMPUTE_ARCH:-x86_64}"

STORE="$PLG_GROUPS_STORAGE/$PLG_GROUP"
HEAVY="$STORE/users/$USER/$PROJECT_NAME"
PROJECT="${CFM_PROJECT_DIR:-$HOME/projects/$PROJECT_NAME}"

test -d "$STORE" || { echo "No such storage group: $STORE" >&2; exit 1; }
test -w "$STORE" || { echo "Storage group is not writable: $STORE" >&2; exit 1; }

umask 077
mkdir -p \
    "$HEAVY/envs" \
    "$HEAVY/local-$COMPUTE_ARCH" \
    "$HEAVY/cache/uv" \
    "$HEAVY/cache/pip" \
    "$HEAVY/cache/torch" \
    "$HEAVY/cache/triton" \
    "$HEAVY/cache/huggingface-hub" \
    "$HEAVY/datasets" \
    "$HEAVY/models" \
    "$HEAVY/outputs"
chmod 700 "$HEAVY"

ensure_link() {
    local link_path="$1" target="$2"

    if [[ -L "$link_path" ]]; then
        [[ "$(readlink "$link_path")" == "$target" ]] && return 0
        echo "Refusing to replace symlink: $link_path -> $(readlink "$link_path")" >&2
        return 1
    fi
    if [[ -e "$link_path" ]]; then
        echo "Refusing to replace existing path: $link_path" >&2
        return 1
    fi
    ln -s "$target" "$link_path"
}

test -d "$PROJECT" || {
    echo "No checkout at $PROJECT. Clone or rsync the repository there first." >&2
    exit 1
}

# Names chosen to match what the repo already reads: conf/dataset/*.yaml points at
# data/, train.py writes outputs/, and uv builds .venv/.
ensure_link "$PROJECT/.local" "$HEAVY/local-$COMPUTE_ARCH"
ensure_link "$PROJECT/.cache" "$HEAVY/cache"
ensure_link "$PROJECT/.venv" "$HEAVY/envs/a100"
ensure_link "$PROJECT/data" "$HEAVY/datasets"
ensure_link "$PROJECT/models" "$HEAVY/models"
ensure_link "$PROJECT/outputs" "$HEAVY/outputs"

echo "PROJECT=$PROJECT"
echo "HEAVY=$HEAVY"
for item in .local .cache .venv data models outputs; do
    printf '  %s -> %s\n' "$item" "$(readlink "$PROJECT/$item")"
done
echo
echo "Next: build the environment in group storage, then dry-run a job."
echo "  export UV_PROJECT_ENVIRONMENT=\"$HEAVY/envs/a100\""
echo "  export UV_CACHE_DIR=\"$HEAVY/cache/uv\""
echo "  cd \"$PROJECT\" && uv sync"
echo "  ./scripts/launch_slurm.sh --gpus 4"

# Set up an environment on PLGrid

## Contents

1. [Select and verify a cluster profile](#select-and-verify-a-cluster-profile)
2. [Configure local SSH access](#configure-local-ssh-access)
3. [Inspect access and allocations](#inspect-access-and-allocations)
4. [Separate code from heavy files](#separate-code-from-heavy-files)
5. [Create project-scoped storage links](#create-project-scoped-storage-links)
6. [Configure caches and Python](#configure-caches-and-python)
7. [Validate the setup](#validate-the-setup)

## Select and verify a cluster profile

The following profiles were verified on 2026-07-22. Always compare them with
live commands because grants, partitions, modules, and service names can
change.

| Item | Helios GH200 | Athena A100 |
|---|---|---|
| SSH host | `login01.helios.cyfronet.pl` | `athena.cyfronet.pl` |
| Compute account | `plgcountercontex-gpu-gh200` | `plgtabcfs-gpu-a100` |
| Partition | `plgrid-gpu-gh200` | `plgrid-gpu-a100` |
| Storage group | read from `hpc-fs` | read from `hpc-fs` |
| Storage service | STORAGE-02 | STORAGE-01 |
| Compute architecture | `aarch64` | `x86_64` |
| Verified software path | `ML-bundle/25.10` | `GCC/11.3.0 OpenMPI/4.1.4 PyTorch/1.13.1-CUDA-11.7.0` |

The Slurm account and storage group can be different identifiers. Read the
actual identifiers from `hpc-grants` and `hpc-fs`; do not derive one from the
other.

Athena and Ares expose the same STORAGE-01 allocation, but their home
directories are separate. Install or synchronize code into the home directory
of the cluster where the job will run.

## Configure local SSH access

Prefer a public key registered with PLGrid. A local `.env` may hold non-secret
selection values and the path to a private key:

```dotenv
PLG_LOGIN=plg_example
PLG_HOST=athena.cyfronet.pl
PLG_IDENTITY_FILE=/absolute/path/to/private_key
PLG_GRANT=plgtabcfs-gpu-a100
PLG_PARTITION=plgrid-gpu-a100
PLG_GROUP=replace-with-hpc-fs-group
```

Protect it with `chmod 600 .env`, ignore it in Git, and never display it while
debugging. Avoid storing `PLG_PASSWORD`; use the SSH key or agent instead. Do
not shell-source an untrusted env file. Read only the expected variables.

Connect with the selected login and host:

```bash
ssh -i "$PLG_IDENTITY_FILE" "$PLG_LOGIN@$PLG_HOST"
```

Omit `-i` when the correct key is already selected by `~/.ssh/config` or an
SSH agent.

## Inspect access and allocations

Run these commands on the login node before preparing storage or jobs:

```bash
hpc-grants
hpc-fs
sinfo -p "$PLG_PARTITION" -o '%P %a %l %D %G'
scontrol show partition "$PLG_PARTITION"
quota -s 2>/dev/null || true
du -sh "$HOME"
module avail 2>&1 | less
```

Select the writable group shown by `hpc-fs`, export its exact identifier, and
verify the resulting path before creating anything:

```bash
export PLG_GROUP="replace-with-hpc-fs-group"
test -d "$PLG_GROUPS_STORAGE/$PLG_GROUP"
test -w "$PLG_GROUPS_STORAGE/$PLG_GROUP"
```

Confirm all of the following:

- The intended compute account is active and appears in `hpc-grants`.
- The intended storage group is active and appears in `hpc-fs`.
- `$PLG_GROUPS_STORAGE/$PLG_GROUP` exists and is writable.
- The requested partition is active and accepts the account.
- The user knows the current home quota; 10 GiB was observed during the
  verified workflows, but the live quota is authoritative.
- The required module still exists. Use `module spider NAME` or the local
  equivalent when a recorded module is unavailable.

Stop before submission if an account, storage allocation, or partition does
not match the user's active grant.

## Separate code from heavy files

Use this layout for each project:

```text
$HOME/projects/<project>/                         code, Slurm files, small config
$PLG_GROUPS_STORAGE/<group>/users/$USER/<project>/
├── envs/                                         Python environments
├── local-<architecture>/                         project-local installs
├── cache/                                        package and model caches
├── datasets/                                     durable input data
├── models/                                       weights and checkpoints
└── outputs/                                      durable results
```

This keeps the home directory small while preserving code where login and
compute nodes can reach it. Use a per-user subdirectory inside team storage to
avoid collisions and accidental sharing.

For I/O-heavy jobs, stage active inputs in `$SCRATCH/$SLURM_JOB_ID` during the
job and copy outputs back to the durable `outputs/` directory. Never leave the
only copy in scratch.

## Create project-scoped storage links

On the target cluster, define the selected names and create directories with a
private default permission mask:

```bash
PROJECT_NAME=my-project
PROJECT="$HOME/projects/$PROJECT_NAME"
HEAVY="$PLG_GROUPS_STORAGE/$PLG_GROUP/users/$USER/$PROJECT_NAME"

umask 077
mkdir -p "$PROJECT" "$HEAVY"/{envs,cache,datasets,models,outputs}
chmod 700 "$HEAVY"
```

Set the compute architecture from the verified profile, not from the login
node. Helios login and compute nodes can have different architectures:

```bash
COMPUTE_ARCH=aarch64  # Use x86_64 for Athena A100.

mkdir -p "$HEAVY/local-$COMPUTE_ARCH" \
  "$HEAVY/cache"/{uv,pip,huggingface-hub,huggingface-datasets,torch}

ln -s "$HEAVY/local-$COMPUTE_ARCH" "$PROJECT/.local"
ln -s "$HEAVY/cache" "$PROJECT/.cache"
ln -s "$HEAVY/envs/default" "$PROJECT/.venv"
ln -s "$HEAVY/datasets" "$PROJECT/data"
ln -s "$HEAVY/models" "$PROJECT/models"
ln -s "$HEAVY/outputs" "$PROJECT/outputs"
```

Before every `ln -s`, check whether the link path already exists. Accept it
only when it is already a symbolic link to the exact intended target. Refuse
to replace files, directories, or links to another target.

Do not link all of `~/.local` or `~/.cache` into team storage. Project-scoped
links prevent unrelated credentials and packages from being exposed and avoid
mixing binary packages for different architectures. This is especially
important on Helios because its login and GH200 compute architectures differ.

## Configure caches and Python

Export durable cache paths in environment-setup and experiment jobs:

```bash
export XDG_CACHE_HOME="$HEAVY/cache"
export UV_CACHE_DIR="$HEAVY/cache/uv"
export PIP_CACHE_DIR="$HEAVY/cache/pip"
export HF_HUB_CACHE="$HEAVY/cache/huggingface-hub"
export HF_DATASETS_CACHE="$HEAVY/cache/huggingface-datasets"
export TORCH_HOME="$HEAVY/cache/torch"
```

Use `uv` for Python dependency management when it is available. Build the
environment at the final group-storage path rather than building it in home
and moving it afterward.

Helios-specific rules:

- Build native environments inside a GH200 compute job so wheels and binaries
  target `aarch64`; the login node may be `x86_64`.
- Load the current ML bundle and inspect its configured package index before
  installing GPU packages.
- The verified example used `ML-bundle/25.10`, `uv`, and PyTorch
  `2.9.0+cu129`; verify current availability.

Athena-specific rules:

- The verified catalog did not publish `uv`. This is a specific reason to use
  `python -m venv --system-site-packages` and inherit the published PyTorch
  module.
- The verified example loaded GCC, OpenMPI, and PyTorch 1.13.1 with CUDA 11.7;
  inspect the current module catalog before reusing those versions.
- If environment creation was interrupted, repair it with
  `python -m venv --upgrade --system-site-packages <env>` and verify
  `include-system-site-packages = true` in `pyvenv.cfg`.

## Validate the setup

Finish setup only after checking:

```bash
test -d "$PROJECT"
test -w "$HEAVY/outputs"
find "$PROJECT" -maxdepth 1 -type l -exec ls -ld {} +
du -sh "$PROJECT" "$HEAVY"
```

Resolve every symlink and confirm it stays within the selected project storage
root. Confirm that `.env` is absent from the remote project unless the user
explicitly needs a remote secret file and has approved that transfer.

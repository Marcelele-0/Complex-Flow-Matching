---
name: wcss-run
description: Run CyFM workloads on the WCSS / KDM cluster (login ui.wcss.pl, Slurm account hpc-danbor2008-1756464546). Use when connecting over SSH, checking the grant balance, working in the shared project storage, writing or submitting Slurm jobs, monitoring or diagnosing them, or fetching results back.
---

# CyFM on WCSS (KDM Wrocław)

Everything here was observed on the machine between 2026-09-07 and 2026-09-12. When
you learn something new, append it here rather than keeping it in your head.

## Read this first: the storage is shared with other groups

The grant tree holds directories owned by at least five other accounts (`domgal7481`,
`julfar7064`, `tymem12`, `mikcza9427`, `annpob7152`) with datasets, checkpoints and
results that are not ours, several of them group-writable. Rules:

- Work inside `$PDDIR/CyFM` (ours) and nowhere else. Never write to the project root.
- Never run a destructive command with a relative path: `pwd` first, read the absolute
  path back, then act.
- Never `rm -rf` with a glob or a variable in that tree. Delete named paths.
- Ask before deleting anything you did not create in this session.

**The `cd` trap.** A bare `cd` does not go to `$HOME` here; it lands in the grant
storage, because `$PDDIR` is set in the environment. `cd ~` behaves normally. A script
of the shape `cd && rm -rf build` therefore deletes from *shared project storage*.
Always `cd` to an explicit absolute path.

## Layout (set up 2026-09-12, group-readable and group-writable)

| path | what | why there |
| --- | --- | --- |
| `$PDDIR/CyFM` | code, `outputs/`, `logs/`, `cache/` | one place the whole group can read, rerun and inspect; setgid so new files keep the grant group |
| `$HOME/.venv-cyfm` | the uv environment, one per user | tens of thousands of small files, which lustre serves badly, and it must not be shared |
| node-local `/tmp` | framework caches (torch, triton, inductor, matplotlib) | keeps home and lustre free of cache churn |

`$PDDIR` expands to `/lustre/pd03/hpc-danbor2008-1756464546`. `scripts/wcss/env.sh`
sets all of this, including `UV_PROJECT_ENVIRONMENT` (which is what lets `uv run
--no-sync` work with the project on lustre and the venv in home) and `umask 0002`, so
files stay group-writable. Source it at the top of every batch script.

## First run as a new user

```bash
ssh <user>@ui.wcss.pl                      # the banner warns about post-quantum KEX; ignore it
mkdir -p $HOME/.local/bin                  # uv is not installed system-wide
# unpack the uv release tarball into ~/.local/bin (the login node has internet)
sbatch $PDDIR/CyFM/scripts/wcss/setup_env.sbatch
```

`setup_env.sbatch` builds the venv **on a compute node** (the login node caps each user
at 1 GB of RAM and the OOM killer is silent), prints the GPU and torch versions, then
trains one 64x64 arm for two epochs so the timing is on record. `uv sync --frozen
--no-dev` took 98 s; compute nodes have internet and uv fetches its own Python 3.11.

## Gotchas paid for once

- **The venv remembers where the project was.** `uv` installs `cyfm` as a path
  dependency, so moving or deleting the checkout leaves `import cfm` failing with
  `ModuleNotFoundError` even though the venv looks intact. Fix: `uv sync --frozen
  --no-dev --offline` from the new location (the deps are already cached, so it is a
  two-second reinstall of the project alone).
- **Renaming a project field's options wipes the values.** Editing the Status options
  of the GitHub project through `updateProjectV2Field` recreated them with new ids and
  cleared every item's Status. Set field values *after* any option rename.
- **`squeue -j <array-id>` can return nothing while tasks still run.** Count states with
  `sacct -j <id> -X -n -o State | sort | uniq -c` instead, or a wait loop exits early.

## Hardware and partitions (verified)

| partition | max wall | nodes | GPUs |
| --- | --- | --- | --- |
| `lem-gpu-short` | 3 days | 74 | 296 |
| `lem-gpu-normal` | 7 days | 52 | 208 |
| `lem-gpu-interactive` | 6 h | 2 | 8 |
| `bem2-cpu-normal` | 21 days | — | — |

GPU nodes: four H100 (95 GB each), driver 595, `storage:local:7000G` of node-local
scratch, `storage:shm:955G`. Stage a dataset to node-local `/tmp` at job start rather
than reading lustre every step.

**The GPU grant is the scarce resource**: ~7.5k GPU-hours left of 14k, against ~45k CPU
hours of 99k. Downloads, extraction and ESPIRiT belong on `bem2-cpu-normal`, not on an
H100. Check with `service-balance` and record the reading here when it moves.

## The sbatch header this site expects

```bash
#SBATCH --partition=lem-gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:hopper:1          # the GPU type must be named; gpu:1 is rejected
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/CyFM/logs/%x-%A_%a.txt
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/CyFM/logs/%x-%A_%a.err
```

No `--account` flag: the grant is the default account. `--output` cannot expand
environment variables, so the log path is absolute. Job arrays are the normal way to
sweep here.

## Running CyFM

```bash
source $PDDIR/CyFM/scripts/wcss/env.sh          # also cds into the project

# one arm: train + evaluate, LOSS_MODE in {l2u, l1u, l1w}
LOSS_MODE=l2u bash scripts/paper/run_arm.sh NAME SIDE GEOMETRY COUPLING SEED [EPOCHS]

# the full paper grid as an array; the header lists the narrowing variables
sbatch scripts/wcss/paper_tables_l2u.sbatch
SEEDS="3 4" sbatch --array=0-23 --export=ALL scripts/wcss/paper_tables_l2u.sbatch
```

Measured throughput on one H100: 64x64 fields at ~5 s/epoch (25 it/s), so a 40-epoch
run plus evaluation is 4-5 minutes; 16x16 and 32x32 are dominated by process start-up.
The same run takes ~20 minutes on the workstation's RTX 4070 Ti SUPER.

Results land in `$PDDIR/CyFM/outputs/{train,evaluate}/<run>/<timestamp>/`; pull them
with `rsync -av --partial`, never by editing in the shared tree.

## Operating rules

- **Before submitting:** write the sbatch file, read it back in full, state the
  requested resources and the estimated GPU-hour cost, and only then submit.
- **Monitoring:** one deliberate check beats a polling loop. `squeue -u $USER` for
  state, `sacct -j <id> -X --format=JobID,State,Elapsed,ExitCode` afterwards. For an
  array, count states with `sacct -j <id> -X -n -o State | sort | uniq -c`; `squeue -j`
  on the array's parent id can return nothing while tasks are still running, which will
  fool a wait loop.
- **Diagnosing:** read the job's own stdout and stderr under `logs/` before theorising.
- **Long jobs:** assume preemption; checkpoint and make the job resumable before asking
  for a large wall clock.
- **Shell gotcha:** the assistant's local shell is zsh, where `for x in $VAR` does not
  word-split. Write explicit lists or use arrays when scripting job submissions.

## Documentation

- Handbook: https://man.e-science.pl/pl/kdm
- Job FAQ: https://man.e-science.pl/pl/kdm/faq
- Usage statistics: https://hpc-info.kdm.wcss.pl (needs the KDM VPN)

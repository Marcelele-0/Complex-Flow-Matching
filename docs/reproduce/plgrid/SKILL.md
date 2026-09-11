---
name: plgrid-run
description: Run Complex Flow Matching training on PLGrid GPU clusters, especially Athena A100 (plgrid-gpu-a100). Use when connecting to PLGrid over SSH, checking grants and storage with hpc-grants/hpc-fs, preparing project storage and environments around the home quota, writing or submitting Slurm jobs, launching multi-GPU DDP training, monitoring or diagnosing a job, or retrieving results.
---

# Run CFM workloads on PLGrid

Procedural guidance, not a tool. There is no controller to invoke here: execute
ordinary SSH, filesystem and Slurm commands as the task requires, and treat every
executable-looking path mentioned below as something to read and adapt before it is
run.

## Scope

This repository's Slurm scripts target **Athena A100 only**
(`plgrid-gpu-a100`, x86_64). `launch_slurm.sh`, `train_ddp.sbatch` and
`bootstrap_plgrid_storage.sh` hardcode that partition and architecture.

The two general references below still describe **Helios GH200** as well, because
they are upstream text. Use them for Helios reasoning, but do not point this repo's
scripts at it: GH200 compute nodes are `aarch64` while the login node may not be, so
the environment has to be built inside a GH200 job and the module set differs. Adding
Helios means a second sbatch and a second bootstrap, not a changed flag.

## Route the task

- Read [references/cfm-on-athena.md](references/cfm-on-athena.md) first for anything
  in **this repository** - launching training, the DDP layout, resume behaviour, what
  to check after a job. It is the concrete instance of the two general references.
- Read [references/setup-environment.md](references/setup-environment.md) in full
  when connecting, checking access, preparing storage, creating Python environments,
  or arranging project directories and symlinks.
- Read [references/running-experiments.md](references/running-experiments.md) in full
  when preparing, submitting, monitoring, troubleshooting or retrieving a Slurm
  experiment in general.

## Apply these rules

1. Prefer an SSH key or SSH agent. Never print, commit or copy `.env` or passwords to
   PLGrid.
2. Inspect live `hpc-grants`, `hpc-fs`, partition and module information before
   relying on a recorded value. The cluster profiles in the references are verified
   examples, not service discovery.
3. Keep code and small configuration in the cluster home directory. Keep
   environments, caches, datasets, models and durable outputs in allocated group
   storage. Home on Athena is a ~10 GiB quota that one torch environment fills.
4. Create project-scoped links such as `project/.venv` and `project/.cache`. Never
   replace global `~/.local` or `~/.cache`.
5. Refuse to overwrite an existing path or an unexpected symbolic link during storage
   setup.
6. Run `sbatch --test-only` before a new job shape. `scripts/launch_slurm.sh` does
   this on every path and needs `--submit` before it consumes allocation. Submit only
   when the user has asked you to.
7. Treat `$SCRATCH` as temporary. Copy durable results back to group storage.
8. Verify jobs with `sacct`, the logs and the expected result files. A returned job ID
   is not evidence of success.

## Do not

- Do not `scancel` a job unless the user asked for it, and name the exact job ID
  before you do.
- Do not resubmit a `PENDING` job because it looks stuck. Read the reason `squeue`
  gives.
- Do not raise `--mem` or `--time` in response to a failure before reading what the
  job actually used in `sacct`.

## Provenance

Adapted from [ofurman/plgrid-skill](https://github.com/ofurman/plgrid-skill)
(`plgrid-run`, commit `7f7fdf1`, verified against PLGrid on 2026-07-22). The two
general references are upstream text, unmodified. `references/cfm-on-athena.md` and
the routing above are specific to this repository. Re-sync by diffing against
upstream; the cluster facts age, the storage discipline does not.

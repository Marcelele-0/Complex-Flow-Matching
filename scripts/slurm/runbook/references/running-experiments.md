# Run experiments on PLGrid

## Contents

1. [Prepare the project](#prepare-the-project)
2. [Define the Slurm request](#define-the-slurm-request)
3. [Use durable storage and scratch](#use-durable-storage-and-scratch)
4. [Validate and submit](#validate-and-submit)
5. [Monitor jobs](#monitor-jobs)
6. [Verify results](#verify-results)
7. [Diagnose failures](#diagnose-failures)

## Prepare the project

Complete the storage and environment checks in
`references/setup-environment.md` before the first experiment in a project.
Keep source code, Slurm files, and small configuration under
`$HOME/projects/<project>`. Keep environments, datasets, models, checkpoints,
and durable output under the project's group-storage root.

Synchronize code without copying secrets or local heavy directories:

```bash
rsync -av \
  --exclude '.env' \
  --exclude '.venv' \
  --exclude '.cache' \
  --exclude '.local' \
  --exclude 'data' \
  --exclude 'models' \
  --exclude 'outputs' \
  --exclude '*.out' \
  --exclude '*.err' \
  ./ "$PLG_LOGIN@$PLG_HOST:projects/<project>/"
```

Review `.gitignore`, generated configuration, and command-line arguments for
secrets before synchronization. Keep reproducibility metadata such as the Git
commit, parameters, module list, and job ID with each experiment output.

## Define the Slurm request

A GPU job should state its resources explicitly:

```bash
#!/bin/bash -l
#SBATCH --job-name=<short-name>
#SBATCH --account=<active-compute-account>
#SBATCH --partition=<active-gpu-partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=<cpus>
#SBATCH --mem=<memory>
#SBATCH --gres=gpu:1
#SBATCH --time=<walltime>
#SBATCH --output=<name>-%j.out
#SBATCH --error=<name>-%j.err

set -euo pipefail
```

Use values reported by `hpc-grants`, `sinfo`, and `scontrol`; do not infer an
account from a storage group. Request only the resources the experiment needs.
Keep wall time short for a smoke test.

Inside the job:

1. Set project and heavy-storage paths explicitly.
2. Export project cache variables.
3. Purge inherited modules and load the verified module set.
4. Change to the project directory.
5. Use `srun` for the experiment process.
6. Write durable output under the storage-backed `outputs/` path.

Use the files under `assets/examples/` as starting points for the verified
Helios GH200 and Athena A100 job shapes.

## Use durable storage and scratch

For small smoke tests, direct reads and writes to group storage are sufficient.
For I/O-heavy work, stage active files to job-local scratch:

```bash
JOB_SCRATCH="$SCRATCH/$SLURM_JOB_ID"
mkdir -p "$JOB_SCRATCH"
rsync -a "$HEAVY/datasets/<input>/" "$JOB_SCRATCH/input/"

# Run the experiment against $JOB_SCRATCH/input and $JOB_SCRATCH/output.

rsync -a "$JOB_SCRATCH/output/" "$HEAVY/outputs/$SLURM_JOB_ID/"
```

Arrange copy-back so failures and signals do not silently discard valuable
checkpoints. Scratch retention policies vary and may change; inspect the
current cluster documentation. Never treat scratch as an archive.

## Validate and submit

For a new or changed job request, validate it without consuming allocation:

```bash
: "${PLG_GROUP:?Set PLG_GROUP to the writable group shown by hpc-fs}"
sbatch --test-only experiment.sbatch
```

Slurm exports the submission environment by default, so the job receives the
selected `PLG_GROUP`. If the site or submission command disables environment
export, pass `PLG_GROUP` explicitly with the appropriate `sbatch --export`
option.

Inspect the command output before submitting. When the user has asked to run
the experiment:

```bash
job_id=$(sbatch --parsable experiment.sbatch)
printf '%s\n' "$job_id" > experiment.jobid
squeue -j "$job_id"
```

For a setup followed by an experiment, use an `afterok` dependency so the
experiment cannot run against a failed or partial environment:

```bash
setup_id=$(sbatch --parsable setup-env.sbatch)
experiment_id=$(sbatch --parsable \
  --dependency="afterok:$setup_id" experiment.sbatch)
```

Keep every returned job ID. Do not interpret successful submission as
successful execution.

## Monitor jobs

Use `squeue` for live scheduler state:

```bash
squeue -j "$job_id" -o '%.18i %.12P %.24j %.10u %.2t %.10M %.6D %R'
```

Use `sacct` for final state and allocated resources:

```bash
sacct -j "$job_id" \
  --format=JobID,JobName,Partition,Account,State,Elapsed,AllocTRES,ReqMem,ExitCode
```

Accounting can lag briefly after a job leaves the queue. Recheck before
declaring the job missing. For dependency chains, inspect every stage rather
than only the last job.

## Verify results

Require all of the following before reporting success:

- The relevant job and step states are `COMPLETED`.
- `ExitCode` is `0:0`.
- Standard error contains no hidden failure.
- The expected durable output exists and is non-empty.
- A GPU smoke test confirms CUDA availability, identifies the expected device,
  executes a real tensor calculation, synchronizes CUDA, and writes a
  structured result.

Example inspection:

```bash
cat "experiment-${job_id}.out"
cat "experiment-${job_id}.err"
test -s "outputs/result-${job_id}.json"
cat "outputs/result-${job_id}.json"
```

Record the Git commit, parameters, hostname, architecture, module list, Python
and framework versions, CUDA version, GPU model, and Slurm job ID when useful
for reproducibility.

## Diagnose failures

Check failures in this order:

1. `sacct` state and exit code.
2. Standard error and standard output.
3. Dependency state for every upstream job.
4. Account and partition compatibility.
5. Storage writability, quota, and symlink targets.
6. Loaded modules and architecture compatibility.
7. Python interpreter path and environment integrity.
8. GPU visibility through `CUDA_VISIBLE_DEVICES`, `nvidia-smi`, and the
   framework runtime.
9. Memory, time, and CPU limits.

Common state interpretations:

- `PENDING`: inspect the reason reported by `squeue`; do not repeatedly submit
  duplicates.
- `OUT_OF_MEMORY`: inspect requested memory and peak use before increasing it.
- `TIMEOUT`: confirm progress and checkpoint behavior before extending time.
- `CANCELLED`: determine who or what cancelled the job.
- `DependencyNeverSatisfied`: inspect the failed upstream job.
- Exit `0:0` with missing output: treat as incomplete and inspect paths and
  copy-back logic.

Cancel a job only when the user requests cancellation or when cancellation is
an explicit part of the active task. Resolve and report the exact job ID before
running `scancel`.

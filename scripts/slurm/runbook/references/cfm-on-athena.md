# Run CFM training on Athena A100

The repository-specific instance of `setup-environment.md` and
`running-experiments.md`. Read those two for the reasoning; this file is what to
actually type for this project.

## Contents

1. [One-time setup](#one-time-setup)
2. [Launch training](#launch-training)
3. [What the job does](#what-the-job-does)
4. [Interruption and resume](#interruption-and-resume)
5. [Verify a run](#verify-a-run)
6. [Diagnose failures](#diagnose-failures)

## One-time setup

Per cluster, on the login node. Confirm every value against live output first -
grants, partitions and modules change, and the recorded ones were verified on
2026-07-22.

```bash
ssh <plg-login>@athena.cyfronet.pl

hpc-grants                       # the compute account, e.g. plg<team>-gpu-a100
hpc-fs                           # the writable storage group (a DIFFERENT identifier)
sinfo -p plgrid-gpu-a100 -o '%P %a %l %D %G'
module spider CUDA               # what the catalog actually publishes today
```

Put the code in home and everything heavy in group storage:

```bash
export PLG_GROUP="<group from hpc-fs>"
mkdir -p "$HOME/projects"
git clone <repo> "$HOME/projects/complex-flow-matching"
cd "$HOME/projects/complex-flow-matching"

./scripts/slurm/bootstrap_plgrid_storage.sh
```

That creates `$PLG_GROUPS_STORAGE/$PLG_GROUP/users/$USER/complex-flow-matching/`
with `envs/ cache/ datasets/ models/ outputs/`, and links `data/`, `models/`,
`outputs/`, `.venv/`, `.cache/`, `.local/` in the checkout to it. It refuses to
replace anything that already exists, so re-running it is safe.

Build the environment **into group storage**, never into home:

```bash
HEAVY="$PLG_GROUPS_STORAGE/$PLG_GROUP/users/$USER/complex-flow-matching"
export UV_PROJECT_ENVIRONMENT="$HEAVY/envs/a100" UV_CACHE_DIR="$HEAVY/cache/uv"
uv sync
```

This project needs `torch>=2.11` and Python 3.11, which is far newer than the
PyTorch module the Athena catalog publishes, so the environment comes from `uv` and
only the CUDA runtime comes from the module system. `uv` is not in the verified
Athena catalog either; install it to `~/.local/bin` if `command -v uv` is empty.

Record the non-secret connection settings once:

```bash
cp .env.example .env && chmod 600 .env   # edit PLG_LOGIN / PLG_GRANT / PLG_GROUP
```

`.env` is git-ignored and must never be copied to the cluster.

## Launch training

`scripts/launch_slurm.sh` resolves the account, builds the resource request,
validates it with `sbatch --test-only`, and only submits when told to.

```bash
# Dry run: validates the job shape, consumes no allocation. This is the default.
./scripts/launch_slurm.sh --gpus 4 -- manifold=cylindrical

# Submit for real.
./scripts/launch_slurm.sh --gpus 4 --submit -e cyl_a100 -- manifold=cylindrical
./scripts/launch_slurm.sh --gpus 4 --submit -e euc_a100 -- manifold=euclidean

# Two nodes, longer wall clock.
./scripts/launch_slurm.sh --nodes 2 --gpus 4 --time 48:00:00 --submit
```

Everything after `--` is a Hydra override forwarded to `train.py`. `--help` lists the
rest of the flags. An Athena A100 node has 8 GPUs; `--gpus 4` is the default because
half a node queues sooner than a whole one.

**`training.batch_size` is per GPU.** The effective batch is
`batch_size * nodes * gpus` and the learning rate is *not* rescaled, so a 4-GPU run
at the default `batch_size=4` is a batch of 16, not 4. Say so when reporting numbers
next to a single-GPU run, or change one of the two.

## What the job does

`scripts/slurm/train_ddp.sbatch` is the job script. In order:

1. `cd` to the submit directory and resolve `$HEAVY` from `PLG_GROUP`. It fails
   immediately if the storage tree is missing.
2. Point every cache (`uv`, `pip`, `torch`, `triton`, HF, W&B) at group storage.
3. `module purge`, then load CUDA only. Override the list with `CFM_MODULES`.
4. Pick a rendezvous host and a port derived from the job ID.
5. `srun` one task per node, each running `torchrun` with `--nproc-per-node` workers.
6. Force `training.auto_resume=true` and pass a preemption sentinel path.

`train.py` then wraps the model in DDP, shards the split with a `DistributedSampler`,
and reduces epoch metrics across ranks so the printed loss is the global one. Only
rank 0 prints, writes checkpoints and talks to W&B.

## Interruption and resume

Two files, two different jobs:

| File | Written | Read by |
|---|---|---|
| `outputs/state/<experiment>/last.pt` | every epoch, atomically | `train.py` on restart |
| `<hydra run dir>/checkpoints/checkpoint_epoch_N.pt` | every 10 epochs | `generate.py`, `evaluate.py` |

`last.pt` carries the weights, the optimizer moments, the cosine schedule position
and the W&B run id. `paths.state_dir` is keyed by `logging.experiment_name` and is
deliberately outside the Hydra run directory, which carries a timestamp a requeued
job would never guess. **Two concurrent runs sharing an `experiment_name` will fight
over that file** - give them different names.

The interruption path:

1. `--signal=B:USR1@600` wakes the batch script 600s before the wall clock.
2. The trap touches `$HEAVY/outputs/preempt-<jobid>`.
3. `train.py` sees the file at the end of the current epoch, agrees across ranks that
   it is time to stop, saves `last.pt` and exits 0.
4. The batch script calls `scontrol requeue`. The requeued job finds `last.pt` and
   continues from the next epoch.

A relayed *signal* cannot do step 3: it would have to survive `srun` -> `uv` ->
the `torchrun` agent, and the agent has no SIGUSR1 handler, so the workers are killed
outright instead of finishing the epoch. Hence the file. If training does not reach
an epoch boundary inside the grace window, a watchdog requeues anyway and `last.pt`
is one epoch stale rather than current.

A job that simply reaches its wall clock ends as `TIMEOUT` and is **not** requeued by
Slurm; that is what the grace signal exists to avoid.

To force a fresh start under an existing experiment name, pass `--no-resume`.

## Verify a run

A submitted job ID is not a result. Require all of:

```bash
squeue -j "$JOB_ID" -o '%.18i %.12P %.24j %.2t %.10M %R'
sacct -j "$JOB_ID" \
  --format=JobID,JobName,Partition,Account,State,Elapsed,AllocTRES,ReqMem,ExitCode

tail -n 50 logs/<job-name>_"$JOB_ID".out
grep -E "Epoch [0-9]+ \|" logs/<job-name>_"$JOB_ID".out | tail -5
ls -la outputs/state/<experiment>/last.pt
```

- state `COMPLETED`, `ExitCode 0:0`
- stderr free of hidden failures
- the epoch line count matches `training.epochs`
- `last.pt` reports `epochs_completed == training.epochs`
- an inference checkpoint exists under the run's `checkpoints/`

A requeued run appears in `sacct` as several rows for one job ID; read all of them.
`SLURM_RESTART_COUNT` is echoed in the log header of each attempt.

## Diagnose failures

In this order, and read before changing anything:

1. `sacct` state and exit code, for **every** attempt of the job ID.
2. stderr, then stdout.
3. `PLG_GROUP` unset or the storage tree missing - the job exits before training with
   an explicit message.
4. Home quota: `du -sh "$HOME"`. A `.venv` that landed in home rather than group
   storage is the usual cause.
5. `torch.cuda.is_available()` false - wrong module, or the environment was built on
   a login node with a different architecture. Athena compute nodes are `x86_64`.
6. A hang with no output for tens of minutes: usually a rank that died while its
   peers wait on a collective. `COLLECTIVE_TIMEOUT` in `src/cfm/utils/distributed.py`
   bounds that at 45 minutes; the first traceback in the logs is the real cause.
7. `OUT_OF_MEMORY`: `batch_size` is per GPU. Check peak use in `sacct` before raising
   `--mem-per-gpu`, and remember `c_unet_cross_slice` runs the encoder on
   `batch_size * num_slices` images.
8. Loss identical across ranks but the run is slower than one GPU: the sampler is not
   sharding. Confirm the "DDP over N ranks" line appears in the log.

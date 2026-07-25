# T02 — Finish HF/FFT experiments and wire W&B logging

**Priority:** P1
**Status:** Todo
**Depends on:** T01 strongly recommended (green tests + clear loss return values)
**Blocks:** fair comparison for T03 metrics

## Goal

1. Run the missing high-frequency boost experiments from `schedule_runs.sh`.
2. Log `loss_hf` and **generated** sample images to Weights & Biases during training.

## Context

Scheduled sweep:

| Run name | Overrides | Local status (`outputs/train/`) |
|----------|-----------|----------------------------------|
| `exp_001_baseline` | default (`lambda_hf=0`) | Missing |
| `exp_002_hf_boost_0.5` | `lambda_hf=0.5` | Done (epoch 100) |
| `exp_003_hf_boost_1.0` | `lambda_hf=1.0` | Missing |
| `exp_004_hf_boost_0.5_factor_6.0` | `lambda_hf=0.5`, `hf_boost_factor=6.0` | Missing |

Today `train.py` logs step/epoch amp & phase losses and a **ground-truth** amplitude image every 10 epochs. It does **not** log `loss_hf` or ODE-generated samples. `loss_hf` is computed inside the criterion but not returned.

`conf/logging/w_and_b.yaml` has `use_wandb: false` by default — enable for these runs.

## What to change

### 1. Return and accumulate `loss_hf`

- Update `DecoupledCylindricalLoss.forward` to return `loss_hf` (e.g. 4-tuple).
- Update `train.py` unpacking, epoch averages, tqdm postfix, and W&B dicts:
  - `step_loss_hf`, `epoch_avg_loss_hf`.

### 2. Log generated samples to W&B

Every N epochs (reuse the existing “every 10 epochs” cadence, or make it configurable):

- Sample noise in cylindrical space (same recipe as `generate.py`).
- Run `CylindricalODESolver` with the current model (eval mode, no grad).
- Convert to complex / magnitude (and optionally phase).
- Log via `wandb.Image` (e.g. `generated_amp`, optionally `generated_phase`).
- Keep GT logging if useful for side-by-side comparison.

Watch GPU memory / time — generation is expensive; keep `num_samples` small (1–4) and `num_steps` reasonable.

### 3. Run the missing experiments

```bash
export WANDB_API_KEY=...
# either full schedule (will re-run exp_002 unless you skip it)
./schedule_runs.sh

# or missing runs only, with W&B on:
uv run src/cfm/train.py logging.use_wandb=true logging.experiment_name=exp_001_baseline
uv run src/cfm/train.py logging.use_wandb=true training.loss.lambda_hf=1.0 logging.experiment_name=exp_003_hf_boost_1.0
uv run src/cfm/train.py logging.use_wandb=true training.loss.lambda_hf=0.5 training.loss.hf_boost_factor=6.0 logging.experiment_name=exp_004_hf_boost_0.5_factor_6.0
```

Optionally re-run `exp_002` with the new logging so all four runs are comparable in W&B.

## Acceptance criteria

- [ ] `loss_hf` visible in W&B for runs with `lambda_hf > 0` (and ~0 when disabled).
- [ ] Generated sample images appear in the W&B dashboard during training.
- [ ] Checkpoints exist for `exp_001`, `exp_003`, `exp_004` through epoch 100 (and `exp_002` if re-run).
- [ ] Short note or W&B report comparing sharpness / artifacts across the four runs (can be a few bullets in the PR).

## Key files

- `src/cfm/flow/torus_math.py`
- `src/cfm/train.py`
- `src/cfm/generate.py` / `src/cfm/flow/solver.py` (reuse generation path)
- `schedule_runs.sh`
- `conf/logging/w_and_b.yaml`
- `conf/training/default.yaml`

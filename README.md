# Complex Flow Matching (CFM) for MRI Synthesis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)

A generative model for synthesizing MRI data using **Flow Matching** on cylindrical manifolds. This work applies continuous normalizing flows to complex-valued MRI images, representing them in amplitude-phase space.

## Motivation

MRI data has unique properties:
- **Complex-valued**: Both magnitude and phase contain clinical information
- **Cylindrical geometry**: Phase lives on a circle $[0, 2\pi)$, amplitude is non-negative

Standard generative models (VAE, diffusion) treat MRI as real-valued Euclidean data, losing these geometric properties. **Flow Matching on cylindrical manifolds** respects the true structure of MRI data, leading to more realistic synthesis.

## Method

### Cylindrical Representation
MRI data is transformed from complex domain to cylindrical coordinates:
```
z = r * e^(iφ)  →  [r, cos(φ), sin(φ)]
```

### Geodesic Flow Bridge
We construct a probability path from pure noise to real MRI data using shortest paths (geodesics) on the cylinder:
- **Amplitude**: Linear interpolation in [0,1]
- **Phase**: Geodesic (shortest angular distance) on the circle

### Training
- Model learns to match a **Continuous Normalizing Flow** to this probability path
- Loss: Decoupled cylindrical loss for amplitude and angular components
- Optimizer: Adam with learning rate scheduling
- Monitoring: Weights & Biases (W&B), opt-in via `logging=w_and_b`

### Generation (Inference)
- Sample random noise in cylindrical space
- Solve ODE from t=0 to t=1 using trained Flow
- Convert back to complex domain and visualize magnitude/phase

### The Euclidean baseline

To know how much the cylindrical geometry actually buys, the repo ships a
**flat R^2 baseline** that runs through the *same* pipeline. Complex pixels are
carried as `[Re z, Im z]`, the bridge is the straight line
`x_t = (1-t)x_0 + t*x_1` with `u = x_1 - x_0`, the loss is a plain L1 over both
channels, and the ODE step is `x <- x + v*dt` with **no phase wrapping, no modulo
arithmetic and no re-projection of any kind**.

Only seven things differ between the two arms - representation, prior, bridge,
loss, solver step, back-transform, and the 2.5D window pipeline. The dataset, the U-Net trunk, the optimizer,
the Heun schedule, the metrics and the checkpoint plumbing are literally shared
code, so a gap in the results can only come from the geometry.

```bash
uv run src/cfm/train.py manifold=euclidean logging.experiment_name=my_baseline
```

## Project Structure

```
src/cfm/
├── train.py              # Training entry point       ) geometry-agnostic:
├── generate.py           # Generation/inference       ) written against the
├── evaluate.py           # Reconstruction evaluation  ) Manifold interface only
├── manifolds/
│   ├── base.py           # The Manifold interface (the six geometric hooks)
│   ├── cylindrical.py    # R^+ x S^1 - adapter over the flow/ classes below
│   └── euclidean.py      # Flat R^2 - the standard flow-matching baseline
├── models/
│   ├── cylindrical_unet.py              # shared trunk; in_channels follows the
│   ├── cylindrical_unet_attention.py    # manifold (3 cylindrical, 2 euclidean)
│   └── cylindrical_unet_cross_slice.py  # 2.5D, cross-slice attention
├── flow/
│   ├── bridge.py            # Geodesic Flow Bridge
│   ├── solver.py            # HeunODESolver base + CylindricalODESolver
│   ├── torus_math.py        # Cylindrical loss functions
│   ├── euclidean_bridge.py  # Straight-line bridge
│   ├── euclidean_solver.py  # Plain Euler step, no projection
│   ├── euclidean_math.py    # Flat velocity loss
│   └── spectral.py          # HF k-space penalty, shared by both losses
├── data/
│   ├── dataset.py        # SKM-TEA dataset loader
│   └── transforms.py     # Data pipelines, one per representation
└── utils/
    ├── complex_ops.py    # Complex <-> cylindrical / euclidean
    ├── inference.py      # Shared model building + checkpoint loading
    └── metrics.py        # PSNR, SSIM, circular phase error

conf/                    # Hydra configuration
├── config.yaml           # Main config
├── hydra/default.yaml    # Output directory setup
├── logging/              # default.yaml (local) vs w_and_b.yaml  <- the toggle
├── manifold/             # cylindrical.yaml vs euclidean.yaml  <- the toggle
├── model/                # Model configs
├── training/             # Training hyperparameters (+ the loss block)
├── dataset/              # Dataset paths
├── generate/             # Generation settings
└── evaluate/             # Evaluation settings

outputs/
├── train/{experiment_name}/{date}_{time}/
│   ├── checkpoints/     # Model weights (.pt)
│   └── wandb/           # W&B logs
├── generate/{experiment_name}/{date}_{time}/
│   └── *.png             # Generated images
└── evaluate/{experiment_name}/{date}_{time}/
    └── metrics.json      # Scored metrics

schedule_runs.sh         # Batch training script (HF sweep)
schedule_comparison.sh   # Trains + scores both geometries for Table 1
```

## Quick Start

### Environment Setup
```bash
git clone https://github.com/Marcelele-0/Complex-Flow-Matching.git
cd Complex-Flow-Matching

uv sync                    # Install dependencies
export WANDB_API_KEY=...   # Only needed if you run with logging=w_and_b
```

### Data

Place your dataset under `data/` (e.g. `data/skm-tea-mini/v1-release`) and point `conf/dataset/skm_tea.yaml` at the correct path. The `data/` directory is git-ignored, so datasets are never committed to the repo.

### Training

**Single run:**
```bash
uv run src/cfm/train.py
```

**Custom experiment:**
```bash
uv run src/cfm/train.py training.loss.lambda_phase=2.0 logging.experiment_name=my_run
```

**Batch schedule:**
```bash
./schedule_runs.sh
```
Edit the script to customize experiment parameters.

**2.5D cross-slice model:**

`c_unet_cross_slice` runs a shared 2D encoder over a window of neighbouring slices,
fuses them with attention at the bottleneck, and decodes the **center slice only**.
It buys volumetric consistency without a full 3D U-Net.

```bash
uv run src/cfm/train.py model=c_unet_cross_slice dataset.num_slices=3
```

`dataset.num_slices` must be odd and greater than 1 for this model; the 2D models
require `num_slices=1`. Training fails immediately on a mismatch rather than
crashing later inside a convolution.

It runs on **either geometry** — the input width follows `manifold.state_channels`
exactly as the 2D trunks do, so the 2.5D row of a comparison has both arms:

```bash
uv run src/cfm/train.py model=c_unet_cross_slice dataset.num_slices=3 manifold=euclidean
```

Normalisation moves behind the stack for slice windows (one peak per window, not
per slice, so inter-slice brightness survives); each manifold supplies that pair
through `build_window_transforms`.

> **Sampling is not wired yet.** The model emits a center-only velocity, which the
> ODE solver cannot use to advance a multi-slice state, so `generate.py` and
> `evaluate.py` raise `NotImplementedError` for it. Use the 2D models to sample.
>
> **Memory:** the encoder runs on `batch_size × num_slices` images, so expect to
> roughly halve `batch_size` versus `c_unet_attention` at `num_slices=3`. Measure
> rather than assume — the bottleneck is actually cheaper here.

### Generation

**Use latest checkpoint from a training run:**
```bash
uv run src/cfm/generate.py generate.run_name=c_unet_attention_run
```

**Custom settings:**
```bash
uv run src/cfm/generate.py generate.run_name=my_run generate.num_samples=10
```

### Evaluation

`generate.py` samples from pure noise, so there is no ground truth to score against.
`evaluate.py` instead measures a **reconstruction**: a real slice is partially noised
via the flow bridge, integrated back to `t=1` by the model, and compared to the
original with PSNR, SSIM, circular phase error, and data consistency error.

```bash
uv run src/cfm/evaluate.py evaluate.run_name=c_unet_attention_run
```

**Key settings:**

| Setting | Meaning |
|---|---|
| `t_start` | How much the model must restore. `0.0` = pure generation, `0.5` = the real test, `1.0` = pipeline self-test (~150 dB) |
| `split` | Which manifest to score (`train`/`val`/`test`), or `null` for every file in `data_dir` |
| `max_samples` | Cap on slices scored; they are strided, not truncated |
| `mask_threshold` | Amplitude floor for the phase error, so air does not dominate |
| `mask.acceleration` | Undersampling factor R for the data consistency error (default 4) |
| `mask.center_fraction` | Fraction of k-space center always sampled (default 0.08) |

```bash
# sweep how much of the reconstruction the model is responsible for
uv run src/cfm/evaluate.py evaluate.run_name=my_run evaluate.t_start=0.75

# quick smoke test: should print PSNR ~150 dB, SSIM 1.0000, phase ~0
uv run src/cfm/evaluate.py evaluate.t_start=1.0 evaluate.num_steps=2 evaluate.max_samples=8
```

Prints a summary table and writes `metrics.json` to the run's output directory. Logs to
W&B when run with `logging=w_and_b`.

> **Training and scoring read the same manifests.** `dataset.split` (default `train`)
> gates what `train.py` loads, and `evaluate.split` (default `test`) gates what is
> scored, both through `cfm/data/splits.py`. With the defaults the scored volumes are
> genuinely held out. Set either to `null` for a directory without `annotations/` —
> the numbers are then reconstruction fidelity on seen data, and must be reported
> as such.

### Side-by-side comparison (cylindrical vs Euclidean)

Both geometries share one pipeline, selected by the `manifold` config group.
Everything except the six geometric hooks is the same code.

**One command, both arms:**
```bash
./schedule_comparison.sh                      # trains, then evaluates, both
TAG=ablation_mse EXTRA="training.loss.vel_loss_type=mse" ./schedule_comparison.sh
```

**Or by hand:**
```bash
uv run src/cfm/train.py    manifold=cylindrical logging.experiment_name=table1_cylindrical
uv run src/cfm/train.py    manifold=euclidean   logging.experiment_name=table1_euclidean

uv run src/cfm/evaluate.py manifold=cylindrical evaluate.run_name=table1_cylindrical
uv run src/cfm/evaluate.py manifold=euclidean   evaluate.run_name=table1_euclidean
```

**Or as a Hydra sweep** (the experiment name interpolates the manifold, so the two
runs cannot collide):
```bash
uv run src/cfm/train.py -m manifold=cylindrical,euclidean \
  'logging.experiment_name=table1_${manifold.name}'
```

| | `manifold=cylindrical` | `manifold=euclidean` |
|---|---|---|
| State | `[m, cos φ, sin φ]` (3 ch) | `[Re z, Im z]` (2 ch) |
| Prior | `m ~ U[0,1]`, `φ ~ U[0,2π)` | same law, drawn without trig — see below |
| Bridge | linear amplitude + geodesic phase | `x_t = (1-t)x₀ + t·x₁`, `u = x₁ - x₀` |
| Loss | `L_amp + λ_φ·L_φ·mask` | one unweighted loss over both channels |
| Solver step | clamp `m ≥ 0`, re-project phase to `R=1` | `x ← x + v·dt`, no constraint at all |
| U-Net | identical trunk — 126 of 128 parameter tensors are element-wise equal under one seed; only `init_conv`'s weight and bias differ | ← |
| Config key | — | `manifold.noise_prior`, `training.loss.vel_loss_type` |

Both write `metrics.json` with a `"manifold"` field, so a results file that has left
its output directory still says which arm produced it.

**Before you publish a number from this table**, two checks:

1. **Run more than one seed per arm.** `training.seed` (default `0`) makes each run
   reproducible *and* puts both arms on element-wise identical weights everywhere
   except `init_conv` — the one module whose shape follows the geometry, built last
   so it cannot shift the RNG for anything else (864 of 73.8M parameters differ).
   Initialisation is therefore not a confound, but data order, the noise draw and
   the time sample still vary, so one run per arm still cannot show a gap exceeds
   that.
   ```bash
   for S in 0 1 2; do
     uv run src/cfm/train.py -m manifold=cylindrical,euclidean training.seed=$S \
       'logging.experiment_name=table1_${manifold.name}_s'$S
   done
   ```
2. **Check `grad_norm` for both arms.** `clip_grad_norm_` is not scale-invariant and
   the two losses differ in magnitude by roughly 4x, so at `training.grad_clip=1.0`
   the clip binds much more often for the cylindrical arm (~67% vs ~33% on synthetic
   data). If it binds very differently on your data, re-run with
   `training.grad_clip=null` and confirm the gap survives — if it does not, the gap
   was about the clip, not the geometry.

**Noise prior.** `manifold.noise_prior` defaults to `uniform`: modulus `~U[0,1]`
with a uniform argument — the *same distribution* the cylindrical arm starts from —
drawn as `a · g/‖g‖` with `a ~ U[0,1]` and `g ~ N(0, I₂)`. An isotropic 2D Gaussian
is rotationally symmetric, so `g/‖g‖` is uniform on the circle without a `cos` or
`sin` anywhere. Both arms therefore transport from one law while the Euclidean path
stays free of trigonometry, and the geometry is the only variable in the table.

Two alternatives are kept as ablations:

```bash
# textbook N(0, I) — what "standard Euclidean flow matching" means in the literature
uv run src/cfm/train.py manifold=euclidean manifold.noise_prior=gaussian

# sample-paired: replays the cylindrical RNG stream so one seed gives both arms the
# identical noise field (tighter still, but reaches it through cos/sin)
uv run src/cfm/train.py manifold=euclidean manifold.noise_prior=matched
```

> **The checkpoint and the manifold must match.** `evaluate.py` builds the model at
> the width the selected manifold declares, so pointing `manifold=euclidean` at a
> cylindrical checkpoint raises a shape error on `init_conv` rather than producing
> plausible-looking, wrong numbers.

**Data consistency error** scores k-space agreement on the sampled trajectories only:
`‖M ⊙ (F(x̂) − F(x))‖²₂ / (‖M ⊙ F(x)‖²₂ + ε)`. The denominator makes it dimensionless
and comparable across slices, resolutions and mask densities — `0.1` means the residual
carries a tenth of the reference's energy on the sampled lines.

> The undersampling mask is **simulated**, not read from the scan — SKM-TEA ships fully
> sampled data. And the reference is `F(x)`, the transform of the ground-truth *image*,
> not raw scanner k-space: the pipeline normalizes amplitude per slice and crops in image
> space, both of which break correspondence with the acquisition. So this measures
> agreement with the reference image on the sampled lines, not fidelity to physical
> measurements.

### Configuration

All settings use **Hydra** in `conf/`:

- **Manifold**: `conf/manifold/` - `cylindrical` or `euclidean`; the geometry toggle
- **Model**: `conf/model/` - UNet architecture, channels, attention
- **Training**: `conf/training/` - batch size, learning rate, epochs, `compile`
- **Loss**: one `training.loss` block serves both geometries. `lambda_phase` /
  `amp_loss_type` / `phase_loss_type` apply to the cylinder, `vel_loss_type` to the
  plane, `lambda_hf` / `hf_boost_factor` to both. Each manifold ignores the keys
  that are not its own, so `training.loss.*` overrides survive the manifold switch.
- **Data**: `conf/dataset/` - dataset path, normalization
- **Logging**: `conf/logging/` - `default.yaml` (local, composed by default) and
  `w_and_b.yaml` (opt in with `logging=w_and_b`). They differ only in `use_wandb`,
  so switching does not move the run's output directory.

## Monitoring

Logging is **opt-in**. A plain run prints to stdout and writes nothing else:

```bash
uv run src/cfm/train.py                     # local only, no W&B
uv run src/cfm/train.py logging=w_and_b     # log to Weights & Biases
```

With `logging=w_and_b` (and `WANDB_API_KEY` exported) a run reports:
- Learning curves (per-step and per-epoch loss, plus the manifold's own breakdown)
- Gradient norms, taken before clipping
- A ground-truth amplitude sample every 10 epochs

View at: https://wandb.ai

## Key Features

✅ **Cylindrical Geometry** - Respects MRI data structure
✅ **Euclidean Baseline** - Same pipeline, flat R^2, for an honest comparison
✅ **Flow Matching** - State-of-the-art generative modeling
✅ **Hydra Configuration** - Reproducible, scriptable experiments
✅ **W&B Integration** - Opt in with `logging=w_and_b`
✅ **Batch Scheduling** - Run multiple experiments sequentially
✅ **Pre-commit Hooks** - Auto-formatting and linting

## Development

### Run tests
```bash
uv run pytest tests/
```

### Format code
Pinned to the version the pre-commit hooks use, so local runs and hooks agree:
```bash
uvx ruff@0.6.9 check --fix src/ tests/   # Linting
uvx ruff@0.6.9 format .                  # Formatting
```

### Pre-commit checks
```bash
pre-commit install         # One-time setup: run hooks on every commit
pre-commit run --all-files
```

`pre-commit install` is per-clone and easy to miss. Until it is run, nothing
enforces ruff, ruff-format or mypy on a commit.

## License

This project is licensed under the [MIT License](LICENSE).

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

### Core Modular Architecture

The repository is built around a highly modular, registry-based architecture. This allows for plug-and-play swapping of Geometries, Neural Architectures, and Reconstructors using Hydra configurations.

```mermaid
graph TD
    %% Define styles
    classDef registry fill:#1f2937,stroke:#3b82f6,stroke-width:2px,color:#fff
    classDef interface fill:#374151,stroke:#10b981,stroke-width:2px,color:#fff
    classDef impl fill:#4b5563,stroke:#6b7280,stroke-width:1px,color:#e5e7eb

    %% Registries
    R_MODELS[MODELS Registry]:::registry
    R_MANIFOLDS[MANIFOLDS Registry]:::registry
    R_SOLVERS[SOLVERS Registry]:::registry
    R_RECONSTRUCTORS[RECONSTRUCTORS Registry]:::registry
    R_DATASETS[DATASETS Registry]:::registry
    R_MASKS[MASKS Registry]:::registry

    %% Interfaces
    I_RECON[BaseReconstructor]:::interface
    I_MANI[BaseManifold]:::interface
    I_SOLV[BaseODESolver / BaseSDESolver]:::interface

    %% Connections from Registries to Interfaces
    R_RECONSTRUCTORS --> I_RECON
    R_MANIFOLDS --> I_MANI
    R_SOLVERS --> I_SOLV

    %% Implementations - Reconstructors
    I_RECON --- Rec_FM[FlowMatchingReconstructor]:::impl
    I_RECON --- Rec_Var[VarNetReconstructor Ceiling]:::impl
    I_RECON --- Rec_Diff[DiffusionReconstructor]:::impl

    %% Implementations - Manifolds
    I_MANI --- Man_Cyl[CylindricalManifold Ours]:::impl
    I_MANI --- Man_Euc[EuclideanManifold Baseline]:::impl
    I_MANI --- Man_Diff[ComplexDiffusionManifold]:::impl

    %% Implementations - Solvers
    I_SOLV --- Solv_Euler[EulerSolver]:::impl
    I_SOLV --- Solv_Heun[HeunSolver]:::impl
    I_SOLV --- Solv_PC[PredictorCorrectorSolver]:::impl

    %% Internal relationships
    Rec_FM -.->|Uses| I_MANI
    Rec_FM -.->|Uses| I_SOLV
    Rec_FM -.->|Uses| R_MODELS

    Rec_Diff -.->|Uses| Man_Diff
    Rec_Diff -.->|Uses| Solv_PC
    Rec_Diff -.->|Uses| R_MODELS

    %% Pipeline flow
    Data[(K-Space Data)] --> R_DATASETS
    R_DATASETS --> R_MASKS
    R_MASKS -->|Undersampled| I_RECON
    I_RECON -->|Reconstructed| Output[(Complex Image)]
```

#### Running Experiments

Thanks to the modular registry and Hydra, comparing methods is as simple as overriding the `manifold`, `model`, or `reconstructor` in the CLI:

**1. Cylindrical Flow Matching (Ours):**
```bash
uv run src/cfm/train.py manifold=cylindrical model=c_unet
```

**2. Euclidean Flow Matching (Baseline):**
```bash
uv run src/cfm/train.py manifold=euclidean model=c_unet
```

**3. Complex Diffusion (Score-MRI Baseline):**
```bash
uv run src/cfm/train.py manifold=complex_diffusion model=c_unet
```

**4. End-to-End Supervised VarNet (Ceiling):**
```bash
uv run src/cfm/evaluate.py model=varnet dataset=fastmri_local
```

You can sweep over these automatically using Hydra's multirun flag (`-m`):
```bash
uv run src/cfm/train.py -m manifold=cylindrical,euclidean,complex_diffusion 'logging.experiment_name=table1_${manifold.name}'
```

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

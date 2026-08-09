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
- Monitoring: Weights & Biases (W&B)

### Generation (Inference)
- Sample random noise in cylindrical space
- Solve ODE from t=0 to t=1 using trained Flow
- Convert back to complex domain and visualize magnitude/phase

## Project Structure

```
src/cfm/
├── train.py              # Training entry point
├── generate.py           # Generation/inference
├── models/
│   ├── cylindrical_unet.py
│   └── cylindrical_unet_attention.py  # U-Net with attention
├── flow/
│   ├── bridge.py         # Geodesic Flow Bridge
│   ├── solver.py         # ODE solver for generation
│   └── torus_math.py     # Cylindrical loss functions
├── data/
│   ├── dataset.py        # SKM-TEA dataset loader
│   └── transforms.py     # Data pipelines
└── utils/
    ├── complex_ops.py    # Complex number utilities
    └── metrics.py        # Evaluation metrics

conf/                    # Hydra configuration
├── config.yaml           # Main config
├── hydra/default.yaml    # Output directory setup
├── logging/w_and_b.yaml  # W&B settings
├── model/                # Model configs
├── training/             # Training hyperparameters
├── dataset/              # Dataset paths
└── generate/             # Generation settings

outputs/
├── train/{experiment_name}/{date}_{time}/
│   ├── checkpoints/     # Model weights (.pt)
│   └── wandb/           # W&B logs
└── generate/{experiment_name}/{date}_{time}/
    └── *.png             # Generated images

schedule_runs.sh         # Batch training script
```

## Quick Start

### Environment Setup
```bash
git clone https://github.com/Marcelele-0/Complex-Flow-Matching.git
cd Complex-Flow-Matching

uv sync                    # Install dependencies
export WANDB_API_KEY=your_key_here  # Add W&B API key
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

### Generation

**Use latest checkpoint from a training run:**
```bash
uv run src/cfm/generate.py generate.run_name=c_unet_attention_run
```

**Custom settings:**
```bash
uv run src/cfm/generate.py generate.run_name=my_run generate.num_samples=10
```

### Configuration

All settings use **Hydra** in `conf/`:

- **Model**: `conf/model/` - UNet architecture, channels, attention
- **Training**: `conf/training/` - batch size, learning rate, epochs
- **Loss**: `lambda_phase` weight for phase component
- **Data**: `conf/dataset/` - dataset path, normalization
- **Logging**: `conf/logging/w_and_b.yaml` - W&B project and experiment names

## Monitoring

All training runs logged to **Weights & Biases**:
- Learning curves (loss, validation metrics)
- Gradient statistics
- Sample generations during training

View at: https://wandb.ai

## Key Features

✅ **Cylindrical Geometry** - Respects MRI data structure
✅ **Flow Matching** - State-of-the-art generative modeling
✅ **Hydra Configuration** - Reproducible, scriptable experiments
✅ **W&B Integration** - Track all runs automatically
✅ **Batch Scheduling** - Run multiple experiments sequentially
✅ **Pre-commit Hooks** - Auto-formatting and linting

## Development

### Run tests
```bash
uv run pytest tests/
```

### Format code
```bash
uv run ruff check --fix .  # Linting
uv run ruff format .        # Formatting
```

### Pre-commit checks
```bash
pre-commit install         # One-time setup: run hooks on every commit
pre-commit run --all-files
```

## License

This project is licensed under the [MIT License](LICENSE).

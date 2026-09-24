# CyFM: Cylindrical Optimal Transport for Few-Step Complex-Valued Flow Matching

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![arXiv](https://img.shields.io/badge/arXiv-Preprint-b31b1b.svg)](https://arxiv.org/)

Code for the paper *CyFM: Cylindrical Optimal Transport for Few-Step Complex-Valued
Flow Matching*.

Complex-valued signals (MRI, audio spectrograms) are usually generated as two real
channels `(Re z, Im z)`. Straight Cartesian paths then pass near the origin, where the
induced angular velocity is unbounded. CyFM runs flow matching on the cylinder
`R+ x S^1` instead, where the phase target is bounded by pi, and couples noise and data
by **exact minibatch optimal transport computed jointly over whole fields in the
cylindrical metric**. The paper measures this across three domains -- synthetic
Gaussian-copula fields with a known ground truth, coil-combined fastMRI knee
acquisitions, and LibriSpeech STFT segments:

- the heavy tail (index ~1) of the Cartesian bridges' angular velocity, network-free,
  on all three targets;
- few-step generation (k <= 8 Heun steps) where CyFM beats the best Cartesian arm on
  synthetic fields at 16x16, 32x32 and 64x64 and on speech spectrograms, with all five
  seeds separated and no significant difference at convergence;
- joint cylindrical OT lowering CyFM's few-step error by 3-60% at every resolution,
  although its transport-cost saving collapses to 3% at 64x64;
- on knee MRI, a 1.8x single-step advantage whose ordering beyond k=1 depends on
  whether the measure sees spatial structure -- reported as measured, including where
  the plane wins;
- the *Factorized Coupling Trap*: coupling per coordinate or per patch keeps every
  marginal exact and destroys the joint distribution.

## Install

```bash
git clone https://github.com/Marcelele-0/Complex-Flow-Matching.git
cd Complex-Flow-Matching
uv sync
```

Python 3.11+, dependencies pinned in `uv.lock`. `uv sync` installs the library and the
development tools; the real-data intake lives behind extras (`--extra mri`,
`--extra audio`, `--extra espirit`) and figure generation behind `--extra viz`.

The network-free results run on a CPU and need no data at all: the synthetic target is
generated on the fly. The synthetic U-Net tables want a CUDA GPU (timings below are on
an RTX 4070 Ti SUPER, 16 GB). The knee MRI and speech tables additionally need their
compact stores built first -- see `scripts/data/`.

## Reproducing the paper

Every result of the paper is a Hydra experiment config in `conf/experiment/` (index:
`conf/experiment/README.md`) whose `paper:` block says how to reproduce it: the
network-free results as commands, the U-Net tables as grids of runs. The U-Net table
configs include their training protocol, so `+experiment=table2_unet64` also trains a
single cell by hand. `scripts/paper/reproduce.py` runs them; a run whose evaluation
already exists is skipped, so a sweep resumes after an interruption and runs shared
between tables are made once.

| what | command | runtime |
| --- | --- | --- |
| every result, in paper order | `uv run python scripts/paper/reproduce.py --all` | ~20 h |
| one result, e.g. Table 2 | `uv run python scripts/paper/reproduce.py table2_unet64` | ~7 h |
| a quick check: one seed at 16x16 | `uv run python scripts/paper/reproduce.py table3_unet_scaling --seeds 0 --sides 16 --tag quick_` | ~15 min |
| what would run | `uv run python scripts/paper/reproduce.py --all --dry-run` | seconds |

The U-Net tables use the objective fixed by the loss ablation (`ablation_loss32`,
`docs/coupling-and-loss.md` section 9): the squared velocity error in both
geometries, with an unweighted phase term on the cylinder, and 5 seeds.

**Without retraining.** Every number in the U-Net tables is archived in
`docs/reproduce/paper_results/`, each with the exact Hydra overrides of its run
(details in that directory's README):

```bash
uv run pytest tests/test_paper_results.py tests/test_experiments.py  # archives == specs
uv run python scripts/paper/paper_tables.py --archive --seeds 0 1 2 3 4 \
    --archive-file docs/reproduce/paper_results/unet_eval_metrics_l2u.json
uv run python scripts/paper/loss_protocols.py --archive   # both loss protocols, p-values
uv run python scripts/paper/latex_tables.py               # the LaTeX the paper includes
```

The arXiv v1 tables (L1 with an amplitude-weighted phase term, 2 seeds) remain
readable with `uv run python scripts/paper/paper_tables.py --archive`.

**Checking the paper's numbers.** `reproducibility/` holds one script per
published result. Each recomputes the number and compares it against the paper,
reporting `PASS`, `FAIL`, or `MISSING INPUT` with the command that would produce
the missing input:

```bash
uv run python -m reproducibility.run_all --network-free   # minutes, CPU, no data
```

That directory's README records the known discrepancies, including one number in
Section 5.4 that is a single draw of a statistic the paper prints as fixed.

**Determinism.** The network-free results (Tables 1 and 4, Sec. 5.3) are seeded through
explicit generators and reprint the paper's numbers exactly. Training is seeded, but GPU
kernels are not bitwise deterministic, so a retrained model matches within the seed
spread rather than to the digit; re-evaluating a fixed checkpoint is exact.

The research log behind the paper -- the gates, sweeps and negative results that did
not make it in -- is `docs/coupling-and-loss.md`.

## Running your own experiment

Training and evaluation are Hydra entry points. The geometry, the coupling and the
dataset are independent switches:

```bash
# CyFM with joint OT on 64x64 synthetic fields
uv run python -m cyfm.train dataset=cylinder_toy_field model=c_unet \
    manifold=cylindrical training.coupling=ot training.bridge=noise \
    logging.experiment_name=my_cyfm

# score it: sliced W2, amplitude W2, circular phase W2, dependence and spatial gaps,
# straightness and angular-velocity probes, at several Heun step counts
uv run python -m cyfm.evaluate dataset=cylinder_toy_field model=c_unet \
    manifold=cylindrical training.coupling=ot training.bridge=noise \
    evaluate.run_name=my_cyfm evaluate.nfe='[1,2,4,8,100]'
```

- `manifold=cylindrical | euclidean` -- the geometry (`conf/manifold/`).
- `training.coupling=independent | ot` -- the coupling. `ot` is one exact assignment over
  the whole batch in the manifold's own metric (`src/cyfm/flow/coupling.py`); no factorised
  variant is offered, deliberately.
- `dataset=cylinder_toy_field | cylinder_toy_iid` -- synthetic copula data with
  `dataset.coupling` (amplitude-phase correlation), `dataset.crop_size` and, for the
  field variant, `dataset.correlation_length`.
- `model=c_unet` (the paper's U-Net) or `model=mlp` (pointwise).
- `manifold.spatial_correlation` / `manifold.phase_spread` -- optional spatially smooth or
  phase-concentrated priors, matched sample for sample across geometries.

## Repository map

```text
src/cyfm/
├── train.py               # training entry point (Hydra); the run is in pipelines/
├── evaluate.py            # evaluation entry point; the sweep is in pipelines/
├── core/                  # the contracts only: BaseManifold, the protocols, the registry
│   ├── protocols.py       # VelocityField, Sampler, Coupling
│   ├── manifold.py        # BaseManifold: what both arms of the comparison share
│   └── pipeline.py        # BasePipeline: setup -> execute -> teardown
├── config/                # the Hydra boundary; nothing outside it sees a DictConfig
│   └── schema.py          # the config sections, parsed once into typed records
├── pipelines/             # the work the two entry points used to hold inline
│   ├── training.py        # TrainingPipeline
│   ├── evaluation.py      # EvaluationPipeline, and the report it renders
│   ├── runtime.py         # loop machinery: reduction, checkpoints, resume, preemption
│   └── reporting.py       # RunLogger: W&B or nothing
├── flow/
│   ├── bridges.py         # geodesic (cylinder) and straight-line (plane) probability paths
│   ├── couplings.py       # independent and joint minibatch-OT couplings
│   ├── losses.py          # the velocity objective in each geometry's own metric
│   ├── solvers.py         # Heun ODE solvers (NFE = 2k - 1) and the VE-SDE baseline
│   └── transport.py       # cost matrices, assignments, W2 estimators
├── manifolds/             # cylindrical (R+ x S^1), euclidean (Re, Im), diffusion baseline
├── models/                # c_unet (the paper's U-Net) and a pointwise MLP control
├── metrics/               # grouped by what each metric can see
│   ├── distributional.py  # pools coefficients: sliced W2, marginal W2, dependence
│   ├── spatial.py         # the nearest neighbour, in amplitude and in phase
│   ├── spectral.py        # every scale at once: the radial power spectrum
│   ├── geometry.py        # the path, not the endpoint: straightness, angular velocity
│   └── summary.py         # the one call the sweep makes
├── data/
│   ├── synthetic.py       # Gaussian-copula target with known amplitude-phase dependence
│   ├── toy.py             # cylinder_toy_iid / cylinder_toy_field datasets
│   ├── transforms.py      # the single place a geometry's pipeline is composed
│   └── stores/            # readers for the prebuilt knee MRI and speech stores
└── utils/                 # complex ops, spectral random fields, seeding, checkpointing

conf/experiment/           # the paper's experiments and training protocols (above)
conf/                      # the other Hydra configs: datasets, manifolds, models, training
scripts/paper/             # reproduce.py and the table / statistics printers
scripts/*.py               # the network-free probes the experiments call
scripts/data/              # builders for the knee MRI and LibriSpeech stores
docs/reproduce/            # archived evaluations with provenance, generated LaTeX tables
docs/coupling-and-loss.md  # why the loss and coupling are what they are
```

## Not part of the paper

- **DDP (`torchrun`)** is supported by `cyfm.train` but was not re-run for the paper:
  every paper result is a single-GPU run.
- **`conf/dataset/fastmri_local.yaml`** points at raw multi-coil k-space and is a
  workstation debug subset. The paper's knee results read the compact store built from
  it by `scripts/data/build_knee_pd_store.py`, not this directory.

## Development

```bash
uv run pytest tests/                    # test suite
uv run ruff check .                     # lint, including Google docstrings
uv run mypy src/ scripts/ reproducibility/   # types
uv run mkdocs build --strict            # API reference (needs --group docs)
pre-commit install                      # ruff, ruff-format and mypy on every commit
```

The API reference is generated from the docstrings, which carry the argument for
a choice rather than a restatement of the signature; `uv run mkdocs serve` reads
it locally.

## Citation

If you use this work or codebase, please cite:

```bibtex
@article{musialek2026cyfm,
  title={CyFM: Cylindrical Optimal Transport for Few-Step Complex-Valued Flow Matching},
  author={Musia{\l}ek, Marcel and Wolanin, Iga and Ryczko, Damian and Grelewska, Anna and Furman, Oleksii},
  journal={arXiv preprint},
  year={2026}
}
```

## License

This project is licensed under the [MIT License](LICENSE).

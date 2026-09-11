# CyFM: Cylindrical Optimal Transport for Few-Step Complex-Valued Flow Matching

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)

Code for the paper *CyFM: Cylindrical Optimal Transport for Few-Step Complex-Valued
Flow Matching* (arXiv preprint, v1).

Complex-valued signals (MRI, audio spectrograms) are usually generated as two real
channels `(Re z, Im z)`. Straight Cartesian paths then pass near the origin, where the
induced angular velocity is unbounded. CyFM runs flow matching on the cylinder
`R+ x S^1` instead, where the phase target is bounded by pi, and couples noise and data
by **exact minibatch optimal transport computed jointly over whole fields in the
cylindrical metric**. The paper measures, on synthetic complex fields with a known
ground truth:

- the heavy tail (index ~1) of the Cartesian bridges' angular velocity, network-free;
- few-step generation (k <= 8 Heun steps) where CyFM beats the best Cartesian arm at
  16x16, 32x32 and 64x64 with all five seeds separated, with no significant difference
  at convergence;
- joint cylindrical OT lowering CyFM's few-step error by 3-60% at every resolution,
  although its transport-cost saving collapses to 3% at 64x64;
- the *Factorized Coupling Trap*: coupling per coordinate or per patch keeps every
  marginal exact and destroys the joint distribution.

## Install

```bash
git clone https://github.com/Marcelele-0/Complex-Flow-Matching.git
cd Complex-Flow-Matching
git checkout feat/generative-task-scope
uv sync
```

Python 3.11+, dependencies pinned in `uv.lock`. The network-free results run on a CPU;
the trained ones want a CUDA GPU (timings below are on an RTX 4070 Ti SUPER, 16 GB).
No dataset download is needed: every result is on synthetic data generated on the fly.

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
`docs/notes/COUPLING_NOTES.md` section 9): the squared velocity error in both
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

**Determinism.** The network-free results (Tables 1 and 4, Sec. 5.3) are seeded through
explicit generators and reprint the paper's numbers exactly. Training is seeded, but GPU
kernels are not bitwise deterministic, so a retrained model matches within the seed
spread rather than to the digit; re-evaluating a fixed checkpoint is exact.

The research log behind the paper -- the gates, sweeps and negative results that did
not make it in -- is `docs/notes/COUPLING_NOTES.md`.

## Running your own experiment

Training and evaluation are Hydra entry points. The geometry, the coupling and the
dataset are independent switches:

```bash
# CyFM with joint OT on 64x64 synthetic fields
uv run python -m cfm.train dataset=cylinder_toy_field model=c_unet \
    manifold=cylindrical training.coupling=ot training.bridge=noise \
    logging.experiment_name=my_cyfm

# score it: sliced W2, amplitude W2, circular phase W2, dependence and spatial gaps,
# straightness and angular-velocity probes, at several Heun step counts
uv run python -m cfm.evaluate dataset=cylinder_toy_field model=c_unet \
    manifold=cylindrical training.coupling=ot training.bridge=noise \
    evaluate.run_name=my_cyfm evaluate.nfe='[1,2,4,8,100]'
```

- `manifold=cylindrical | euclidean` -- the geometry (`conf/manifold/`).
- `training.coupling=independent | ot` -- the coupling. `ot` is one exact assignment over
  the whole batch in the manifold's own metric (`src/cfm/flow/coupling.py`); no factorised
  variant is offered, deliberately.
- `dataset=cylinder_toy_field | cylinder_toy_iid` -- synthetic copula data with
  `dataset.coupling` (amplitude-phase correlation), `dataset.crop_size` and, for the
  field variant, `dataset.correlation_length`.
- `model=c_unet` (the paper's U-Net) or `model=mlp` (pointwise).
- `manifold.spatial_correlation` / `manifold.phase_spread` -- optional spatially smooth or
  phase-concentrated priors, matched sample for sample across geometries.

## Repository map

```text
src/cfm/
├── train.py               # training entry point (Hydra)
├── evaluate.py            # generative evaluation: distributional metrics vs. solver steps
├── flow/
│   ├── coupling.py        # independent and joint minibatch-OT couplings
│   ├── optimal_transport.py  # cost matrices, assignments, W2 estimators
│   ├── solver.py          # Heun / Euler ODE solvers (NFE = 2k - 1 for Heun)
│   └── torus_math.py      # geodesics on the cylinder, decoupled loss
├── manifolds/             # cylindrical (R+ x S^1) and euclidean (Re, Im)
├── models/                # c_unet (+ attention, cross-slice variants), pointwise MLP
├── data/
│   ├── synthetic.py       # Gaussian-copula target with known amplitude-phase dependence
│   ├── toy_dataset.py     # cylinder_toy_iid / cylinder_toy_field datasets
│   └── fastmri.py, ...    # real-data loaders (not used in the paper; see below)
└── utils/random_fields.py # spectral smoothing that keeps every entry N(0, 1)

conf/experiment/           # the paper's experiments and training protocols (above)
conf/                      # the other Hydra configs: datasets, manifolds, models, training
scripts/paper/             # reproduce.py and the table / statistics printers
scripts/*.py               # the network-free probes the experiments call
docs/reproduce/            # archived evaluations with provenance, generated LaTeX tables
docs/notes/                # the research log (COUPLING_NOTES.md)
```

## Not part of the paper

- **fastMRI / SKM-TEA.** `conf/dataset/fastmri_*.yaml` and the loaders are kept for the
  next version (real-data evaluation). They are not exercised by the paper and have not
  been re-validated since the move from reconstruction to generation.
- **DDP (`torchrun`)** is supported by `cfm.train` but was not re-run for the paper:
  every paper result is a single-GPU run.

## Development

```bash
uv run pytest tests/          # test suite
uv run ruff check .           # lint
uv run mypy src/              # types
pre-commit install            # ruff, ruff-format and mypy on every commit
```

## License

This project is licensed under the [MIT License](LICENSE).

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
- few-step generation (k <= 4 Heun steps) where CyFM beats the best Cartesian arm at
  16x16, 32x32 and 64x64, while the Cartesian geometry is better at convergence on the
  larger fields;
- joint cylindrical OT lowering CyFM's few-step error by 11-44% at every resolution,
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

One script per result. Each script's header states the exact configuration and the
numbers it should print.

| paper result | command | runtime |
| --- | --- | --- |
| Table 1 -- bridge angular velocity | `bash scripts/paper/table1_bridges.sh` | ~15 s |
| Table 2 -- U-Net, 64x64, 2 geometries x 2 couplings | `bash scripts/paper/table2_unet64.sh LOGDIR` | ~2.7 h |
| Table 3 -- U-Net at 16x16 and 32x32 (+ the 64x64 runs) | `bash scripts/paper/table3_unet_sizes.sh LOGDIR` | ~40 min |
| Table 4 and Sec. 5.4 -- Factorized Coupling Trap, patch seams | `bash scripts/paper/table4_factorized.sh` | ~4 min |
| Sec. 5.3 -- OT cost saving against field size (86% -> 3%) | `bash scripts/paper/ot_cost_vs_dimension.sh` | ~10 s |
| Tables 2-3 and the per-seed claims of Sec. 5, printed | `uv run python scripts/paper/paper_tables.py` | seconds |

`paper_tables.py` prints Tables 2 and 3 plus the statements that rest on per-seed
evidence: cylinder + OT against the best Cartesian arm at every size and step count,
OT against independent pairing on the cylinder (11-44%, seeds separated in 7 of 9
cells), OT on the Cartesian arm at 16x16 (0.421 -> 0.185), and the straightness of every
64x64 arm.

**Without retraining.** The evaluations the paper was written from are archived in
`docs/paper_results/unet_eval_metrics.json`:

```bash
uv run python scripts/paper/paper_tables.py --archive
```

**Determinism.** The network-free scripts (Tables 1 and 4, Sec. 5.3) are seeded through
explicit generators and reprint the paper's numbers exactly. Training is seeded, but GPU
kernels are not bitwise deterministic, so a retrained model matches within the seed
spread rather than to the digit; re-evaluating a fixed checkpoint is exact.

**Quick end-to-end check** (minutes, not the paper's numbers). The `smoke_` prefix keeps
the short runs from shadowing real ones:

```bash
PREFIX=smoke_ EPOCHS=1 SIZE=128 FIELDS=8 SEEDS=0 SIDES=16 \
    bash scripts/paper/table3_unet_sizes.sh /tmp/cyfm_smoke
```

The full research log behind the paper -- including the gates, sweeps and negative
results that did not make it in -- is `docs/COUPLING_NOTES.md`, with raw console output
in `docs/gate_artefacts/` and its own reproduction index in section 7.

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

scripts/paper/             # one script per paper result (above)
scripts/sweeps/            # the exploratory sweeps behind docs/COUPLING_NOTES.md
scripts/*.py               # network-free gates and probes the paper scripts call
conf/                      # Hydra configs
docs/COUPLING_NOTES.md     # research log, docs/gate_artefacts/ raw outputs
docs/paper_results/        # archived evaluations behind Tables 2 and 3
```

## Not part of the paper

- **fastMRI / SKM-TEA.** `conf/dataset/fastmri_*.yaml` and the loaders are kept for the
  next version (real-data evaluation). They are not exercised by the paper and have not
  been re-validated since the move from reconstruction to generation.
- **`cfmri-suite --eval`** and `schedule_runs.sh` predate that move; `--eval` calls the
  removed reconstruction evaluator and the VarNet baseline, so it does not run on this
  branch. DDP (`torchrun`) and the PLGrid Slurm wrappers (`scripts/launch_slurm.sh`,
  runbook in `docs/plgrid/SKILL.md`) did not change with it, but were not re-run for the
  paper: every paper result is single-GPU.
- `manifold=complex_diffusion` is a variance-exploding SDE baseline, not evaluated here.

## Development

```bash
uv run pytest tests/          # test suite
uv run ruff check .           # lint
uv run mypy src/              # types
pre-commit install            # ruff, ruff-format and mypy on every commit
```

## License

This project is licensed under the [MIT License](LICENSE).

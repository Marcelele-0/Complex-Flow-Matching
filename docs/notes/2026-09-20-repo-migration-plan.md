# Migrating the repository: the plan, and the prompt for the agent that does it

Written 2026-09-20, after the `cfm` -> `cyfm` rename (`b5ca3dd`). Two separate moves are
described here and they must not be confused: **the cluster resync**, which is operational
and can happen any time, and **the clean-repo rewrite**, which happens after the ICLR
submission on 2026-09-25.

## Move 1 --- the cluster tree. Operational, do it before the next job.

`/lustre/pd03/hpc-danbor2008-1756464546/CyFM` is an **rsync'd copy, not a git checkout**,
so it cannot pull the rename. It still holds `src/cfm` and `python -m cfm.*` in five
sbatch files (`bnd_mri`, `bounded`, `dense_nfe_sweep`, `rescore_metrics`,
`table5_fastmri`).

Two traps, both cheap to avoid and expensive to hit:

1. **rsync without `--delete` leaves `src/cfm` in place beside `src/cyfm`.** Python would
   then import whichever the venv's editable install points at, and the stale tree would
   win silently. Remove the old package directory explicitly rather than trusting a flag.
2. **The venv holds an editable install pointing at the old path.** `uv` installs the
   project as a path dependency, so after the move `import cyfm` fails with
   `ModuleNotFoundError` even though the venv looks intact. Fix with
   `uv sync --frozen --no-dev --offline` from the tree (deps are cached; it is a
   two-second reinstall of the project alone). This is the gotcha already recorded in
   the `wcss-run` skill.

Also preserve the setgid shared tree's group-write bits: rsync with
`--no-perms --no-owner --no-group`, as every transfer in this project does.

Nothing under `outputs/` is rewritten. Those `.hydra` configs and `train.log` files record
what was launched at the time, under the name it had then; editing them would falsify the
provenance the archives depend on.

## Move 2 --- the clean repository. After the submission.

### Goal

One package, `cyfm`, that someone can `pip install` and import for the method: manifolds,
bridges, solvers, couplings and the structural metrics. Reproduction of the paper stays in
the repository, pinned at a tag, and depends on the package rather than duplicating it.

### What the survey already established

* **The SDK boundary is nearly where it needs to be.** `omegaconf`/`hydra` appear in
  exactly six files under `src/cyfm/`: three factories (`manifolds/__init__`,
  `data/__init__`, `utils/inference`), two entry points (`train.py`, `evaluate.py`) and
  `utils/config.py`. No manifold, bridge, solver, loss or metric class imports either;
  `manifolds/cylindrical.py` and `utils/metrics.py` import torch and internal modules only.
* **The work is therefore one pattern, six times:** `build_manifold(cfg)` becomes a
  constructor taking plain Python arguments plus a thin `from_config(cfg)` adapter on the
  Hydra side. Same for `build_model` and `build_dataset`.
* **One layering inversion to fix while there:** `utils/metrics.py` imports
  `cyfm.data.synthetic.circular_linear_correlation`. A metrics module reaching into the
  dataset package would drag the synthetic data generator into the SDK's import graph.
* **Packaging decisions are already taken** (see the memory `cfm-pypi-name-cyfm`): the name
  is `cyfm`, checked free on PyPI; discoverability goes in `description` and `keywords`, not
  in alias packages; nothing is published merely to reserve the name. Reproduction is
  addressed by a **tag**, not a long-lived branch, because reproduction needs an immutable
  pointer. The anonymous supplementary bundle is **generated** (`.gitignore` already carries
  `anonymous_submission_build/`), not maintained as a branch: a branch would carry the git
  history, which is the one thing an anonymous bundle must not contain.

### The invariant that governs everything

`tests/test_paper_results.py` is not a normal test file. It checks that **composing the
Hydra overrides recorded in the archives reproduces the configuration of the documented
experiment**. That is the paper's reproducibility claim expressed as code, and it is the
only regression signal the rewrite will have --- the paper's numbers are the specification.

Consequences the migration must respect:

* Renaming a key in `conf/`, or an experiment file, silently breaks the claim. The test
  catches it **only as long as nobody adapts the test to the new shape**. If it must
  change, the numbers have to be re-measured, not the assertions relaxed.
* The archives (`docs/reproduce/paper_results/*.json`) are frozen inputs. They record
  overrides, not module paths, so the rename did not touch them and the rewrite should not
  either.
* The 64x64 knee archive has a shape that does not fit a naive `(side, geometry, coupling,
  seed)` grid: two sweeps of the same checkpoints under different NFE lists, and two arms
  sharing one checkpoint under different sampling regimes. Any new registry must carry that
  shape rather than force the simpler one.

### Ordering

1. Tag the submission commit first. The rebuttal is in January and the current code must
   stay runnable.
2. Branch from `main`. Keep `develop` untouched until the rewrite passes the full suite.
3. Split the library from the Hydra layer (the six files above), one factory at a time,
   full suite green after each.
4. Fix the metrics/data layering inversion.
5. Registries, base classes, mixins --- the part that is actually design work, and the part
   that is safe only once 3 and 4 hold.
6. Dependency split: the SDK's runtime dependencies are torch and little else; `h5py`,
   `nibabel`, `huggingface-hub`, `wandb`, `torchvision`, `pandas`, `soundfile`, `scipy`
   move behind extras. The `fastmri` and `espirit` extras already demonstrate the pattern.
7. Only then, the first PyPI release.

### Verification, after every step

```bash
uv run pytest tests/ -q          # 722 passing as of aadf412
uv run ruff check .
uv run mypy src/
uv run python -m cyfm.train --help
uv run python scripts/prior_control.py --side 16 --num-fields 16   # 0.1841, see below
```

The last one is the cheapest end-to-end check in the repository: it exercises the dataset
registry, the manifold, the prior and the metrics without a network or a checkpoint.

**Correction, 2026-09-20 (`refactor/pypi-architecture`): the exact digits are
platform-dependent and `0.1841` does not reproduce everywhere.** Measured `0.1852` on
macOS 15.6 arm64 with torch 2.11.0 / numpy 2.4.4 / scipy 1.17.1 -- and measured on a
pristine checkout of `775710d`, so it is not a regression introduced by any later work.
The two values differ by 0.6%, which is the order float32 reduction ordering accounts for
between BLAS builds. Treat this gate as a determinism-and-band check (two runs must agree
exactly, and the value must stay near 0.185), not as a literal to grep for. The value is
quoted in this file only; no test asserts it.

Two known-weak spots in that gate, worth fixing early in the rewrite rather than trusting:

* **The pre-commit `mypy` hook runs in its own environment without the project's
  dependencies**, so it sees `DictConfig` and torch types as `Any`. Under full types,
  `tests/` currently has 7 pre-existing errors. `src/` is clean either way.
* **The pre-commit `ruff` is pinned at v0.6.9 while the project uses 0.16.5**, and the two
  disagree on the layout of one `assert` in `scripts/bridge_angular_velocity.py`. The file
  flips back and forth between the hook and a local `ruff format`.

## The prompt

Hand the agent the text below, in a checkout of this repository.

---

> You are migrating the CyFM repository to a clean structure. Read
> `docs/notes/2026-09-20-repo-migration-plan.md` first and follow its ordering; this
> prompt states the constraints, that file states the plan.
>
> **Goal.** One installable package `cyfm` carrying the method --- manifolds, bridges,
> solvers, couplings and the structural metrics --- with the Hydra configs, dataset
> builders, cluster scripts, archives and table tooling left in the repository as the
> reproduction layer. The library must not import `hydra` or `omegaconf`.
>
> **Non-goals.** Do not change any numerical behaviour. Do not rename anything under
> `conf/`. Do not touch `docs/reproduce/paper_results/*.json` or anything under `outputs/`.
> Do not publish to PyPI. Do not rewrite the paper.
>
> **The invariant.** `tests/test_paper_results.py` checks that the Hydra overrides recorded
> in the archives compose to the configuration of the documented experiment. It is the
> paper's reproducibility claim as code and the only regression signal you have, because
> the paper's published numbers are the specification. It must stay green at every step. If
> you believe it has to change, stop and report rather than adapting the assertions --- an
> adapted assertion silently converts a broken claim into a passing test.
>
> **What the survey found, so you need not repeat it.** `omegaconf`/`hydra` appear in six
> files under `src/cyfm/`: `manifolds/__init__.py`, `data/__init__.py`,
> `utils/inference.py`, `train.py`, `evaluate.py`, `utils/config.py`. No manifold, bridge,
> solver, loss or metric class imports either. The work is one pattern applied six times:
> a constructor taking plain Python arguments, plus a thin `from_config(cfg)` adapter on
> the Hydra side. Separately, `utils/metrics.py` imports
> `cyfm.data.synthetic.circular_linear_correlation`, which is a layering inversion --- move
> that function down rather than letting the SDK depend on the dataset package.
>
> **Method.** One change at a time, full verification after each, a commit per step with a
> message saying what moved and why. After every step:
>
> ```bash
> uv run pytest tests/ -q && uv run ruff check . && uv run mypy src/
> uv run python -m cyfm.train --help
> uv run python scripts/prior_control.py --side 16 --num-fields 16   # platform-dependent
> ```
>
> **Report back** with: what moved, what the import surface of `cyfm` now is, which
> dependencies became extras, anything you had to leave alone and why, and any place where
> the existing code was wrong rather than merely untidy. If a step would require changing a
> config key, an experiment file, or an archive, stop and say so instead of doing it.

---

Not in scope for that agent, and deliberately so: the PyPI upload, the anonymous bundle
script, and the P4 non-measurement mask. The first two are one-way doors; the third is an
experiment, not a refactor.

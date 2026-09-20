# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `src/cyfm/py.typed` (PEP 561), so type information actually reaches downstream
  users. Without it `mypy` resolved the whole package to `Any`.
- Package metadata required for a release: `authors`, SPDX `license`, `classifiers`
  and `[project.urls]`.
- `CITATION.cff` and this changelog.
- Continuous integration running the test suite, `ruff` and `mypy`. Until now the only
  workflow moved GitHub Projects issue statuses; nothing gated the test suite, which is
  the repository's sole reproducibility check.

### Changed

- Split the dependencies. The wheel now requires `torch`, `numpy`, `scipy`, plus four
  packages the still-eager import graph forces (`h5py`, `hydra-core`, `tqdm`,
  `python-dotenv`). Everything else moved behind extras (`mri`, `fastmri`, `espirit`,
  `audio`, `logging`, `viz`) or into dependency groups.
- Moved the Jupyter stack into its own `notebooks` group: the jupytext pre-commit hooks
  need it, the test suite does not.

### Removed

Roughly 5,900 lines of code left over from the project's earlier framing as an MRI
reconstruction problem. None of it backed a number in the paper.

- **SKM-TEA and the undersampling masks.** `SKMTEADataset` carried a
  `mode="reconstruction"` branch no config selected, and `masks.py` was reachable only
  through it. With them: the `MASKS` registry, `conf/dataset/skm_tea.yaml`, the
  SKM-TEA half of the downloader, and three test modules. `DEFAULT_DATASET` was
  `"skm_tea"`; it is now `"cylinder_toy_field"`.
- **The paired-Wilcoxon statistics module** (`cyfm/eval/`). It merged evaluation frames
  on `sample_id` -- the statistic of a reconstruction comparison. The paper reports an
  unpaired exact Mann-Whitney U over five seeds, computed in `scripts/paper/`.
- **The dead `LOSSES` layer** (`cyfm/core/loss.py`). Registered under four keys and
  never built. It was also the only reason `core` imported `flow`, so removing it
  closes the `flow -> core -> flow` import cycle that lazy in-method imports had been
  working around.
- **The high-frequency k-space penalty** (`cyfm/flow/spectral.py`, `lambda_hf`,
  `hf_boost_factor`). Disabled by default since it was added; no experiment config
  enables it. The losses lose a return value with it.
- **The attention and cross-slice architectures.** `c_unet_cross_slice` could train but
  never sample, and the guard meant to prevent sampling from it was called by nothing
  but its own test. `c_unet_attention` was unused yet was the root config's default
  model.
- **All four notebooks**, two of which could not run (one imports a function that does
  not exist, one hardcodes another machine's absolute path), plus `schedule_runs.sh`
  and `scripts/clamp_probe.py`, which hardcodes `/home/marcel/...`.
- `manifolds/base.py`, a back-compatibility alias. Its docstring -- the argument for
  why the two-arm comparison is trustworthy -- moved onto the interface it described.
- `torchvision` and `torchdiffeq` from the dependencies. Neither is imported anywhere
  in the repository.
- `plotly` from the development group, for the same reason.
- **The `manim` figure pipeline**: `manim/scenes/flow_viz.py`, which draws the two
  panels of the teaser schematic, `manim/scripts/crop_figures.py`, which trims and
  size-matches them, and both `manim.cfg` files. With them go the `manim` and `pillow`
  dependencies; `manim` pulls in `pycairo`, which needs a system `cairo` and
  `pkg-config`, so its removal is also what lets the project install on a machine that
  has neither. The rendered panels themselves stay: `paper/ICLR Main/figures/`
  continues to hold `EuclideanFlow_cropped.png` and `CylindricalFlow_cropped.png`, so
  the paper still builds, but Figure 1 no longer has a generator in this repository and
  can only be edited as an image.

### Fixed

- `cyfm/flow/__init__.py` declared four names in `__all__` that it never imported, so
  `from cyfm.flow import build_coupling` raised `ImportError` and the `COUPLINGS`
  registry stayed empty unless something imported `cyfm.flow.coupling` directly.
- `check_ascii.py` listed fifteen paths, three of them long deleted, and printed read
  errors without setting its failure flag -- so it exited 0 while checking almost
  nothing. It now walks the tree and treats an unreadable file as a failure.
- `README.md` claimed "every result is on synthetic data generated on the fly" and
  filed fastMRI under "Not part of the paper", while the paper rests on three domains.
  `conf/experiment/README.md` omitted both real-data experiments, and
  `docs/reproduce/paper_results/README.md` omitted both knee archives.
- Dropped two `ruff` exclusions naming `scripts/generate_figure2*.py` and
  `scripts/glue_figure2*.py`; neither file exists.
- Pinned the `ruff` pre-commit hook to v0.16.5, the version the project resolves. The
  hook sat at v0.6.9, and the two disagreed on the layout of one `assert` in
  `scripts/bridge_angular_velocity.py` and one in `tests/test_data/test_build_dataset.py`,
  so both files flipped back and forth depending on who formatted last. Applied the
  formatter once so the tree is consistent; the change is message layout only.
- Replaced the `mypy` pre-commit hook with one that runs in the project environment.
  `mirrors-mypy` installs into an isolated environment without the project's
  dependencies, so it resolved `DictConfig` and every torch type to `Any`.
- Corrected the prior-control gate recorded in
  `docs/notes/2026-09-20-repo-migration-plan.md`. It quoted `0.1841` as the expected
  value; the measured value is `0.1852` on macOS 15.6 arm64, reproduced on a pristine
  checkout of `775710d`. The digits are platform-dependent, so CI asserts determinism
  and a band rather than a literal.

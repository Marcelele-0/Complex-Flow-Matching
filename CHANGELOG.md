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
- Moved `manim` out of the default development group into the `viz` extra. It pulls in
  `pycairo`, which needs a system `cairo` and `pkg-config`; on a machine without them
  `uv sync` failed outright, so the project could not be installed at all. `manim` is
  imported nowhere outside `manim/`.
- Moved the Jupyter stack into its own `notebooks` group: the jupytext pre-commit hooks
  need it, the test suite does not.

### Removed

- `torchvision` and `torchdiffeq` from the dependencies. Neither is imported anywhere
  in the repository.
- `plotly` from the development group, for the same reason.

### Fixed

- Declared `pillow`, which `manim/scripts/crop_figures.py` imports and which was
  previously pulled in only transitively through `matplotlib`.
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

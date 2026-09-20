"""A bare ``import cyfm`` must fill every registry.

These run in a *fresh interpreter*, which is the whole point. Inside the test
session a dozen modules have already been imported, so the registries are full no
matter what ``cyfm/__init__.py`` does, and a test that merely asserted on them
here would pass while the installed wheel was broken.

That is not hypothetical. ``MODELS`` was empty after ``import cyfm`` and filled
only as a side effect of importing :mod:`cyfm.utils.inference`; ``DATASETS``
filled only because a geometry imported ``cyfm.data.transforms``, and emptied the
moment that import was removed. Neither was caught, because
``tests/test_core/test_registry.py`` imports eight modules by hand before it
looks.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

_EXPECTED = {
    "manifolds": {"complex_diffusion", "cylindrical", "euclidean"},
    "models": {"c_unet", "cylindrical_unet", "mlp", "pointwise_mlp"},
    "solvers": {
        "complex_diffusion",
        "cylindrical",
        "cylindrical_heun",
        "cylindrical_ode",
        "euclidean",
        "euclidean_heun",
        "euclidean_ode",
        "pc_diffusion",
    },
    "datasets": {
        "cylinder_toy_field",
        "cylinder_toy_iid",
        "fast_mri",
        "fastmri",
        "fastmri_knee_pd",
        "knee_store",
        "librispeech_stft",
        "stft_store",
    },
    "couplings": {"independent", "none", "optimal_transport", "ot"},
}

_PROBE = """
import json
import cyfm
from cyfm.core.registry import COUPLINGS, DATASETS, MANIFOLDS, MODELS, SOLVERS

print(json.dumps({r.name: r.list() for r in (MANIFOLDS, MODELS, SOLVERS, DATASETS, COUPLINGS)}))
"""


@pytest.fixture(scope="module")
def registries_after_bare_import() -> dict[str, list[str]]:
    """Registry contents in a subprocess that imported nothing but ``cyfm``."""
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, check=True
    )
    return dict(json.loads(completed.stdout))


@pytest.mark.parametrize("registry", sorted(_EXPECTED))
def test_registry_is_populated_by_a_bare_import(
    registry: str, registries_after_bare_import: dict[str, list[str]]
) -> None:
    assert set(registries_after_bare_import[registry]) == _EXPECTED[registry]


def test_every_registry_is_covered_here() -> None:
    """A new registry must be added to the expectations above, not forgotten.

    Without this, adding a sixth registry would silently go unpopulated and
    untested: the parametrisation only walks the keys already listed.
    """
    from cyfm.core import registry as registry_module
    from cyfm.core.registry import Registry

    live = {value.name for value in vars(registry_module).values() if isinstance(value, Registry)}
    assert live == set(_EXPECTED)


def test_the_package_reports_a_version() -> None:
    """``__version__`` is part of what a published package owes its users."""
    import cyfm

    assert isinstance(cyfm.__version__, str)
    assert cyfm.__version__

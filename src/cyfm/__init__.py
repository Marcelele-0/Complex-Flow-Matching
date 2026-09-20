"""CyFM: cylindrical flow matching for complex-valued fields.

Importing this package populates every registry. That is a deliberate cost, paid
here rather than left to whichever module happened to be imported first.

It used to be left. ``MODELS`` was filled only because
:mod:`cyfm.utils.inference` imported :mod:`cyfm.models` for its own use, and
``DATASETS`` only because a geometry imported ``cyfm.data.transforms`` and
dragged the whole data package in behind it. Both worked by accident, neither was
visible from here, and the second stopped working the moment that geometry no
longer needed a transform. A user who installed the wheel and wrote
``cyfm.MODELS.build("c_unet")`` got a ``KeyError``.

The eager imports below cost one import of ``torch``, ``numpy`` and ``h5py``,
which the wheel already requires. Anything behind an extra -- ``soundfile``,
``sigpy``, ``fastmri`` -- stays out of this graph and is imported where it is used.
"""

from importlib.metadata import PackageNotFoundError, version

# Imported for their side effects: @register_* runs at class definition time, so
# a registry holds nothing until the module defining its entries has been
# imported. Listed explicitly, one line per registry, so the population is a
# property of this file rather than of an import order elsewhere.
from cyfm import (
    core,
    data as data,  # noqa: F401  -> DATASETS
    flow as flow,  # noqa: F401  -> COUPLINGS
    manifolds as manifolds,  # noqa: F401  -> MANIFOLDS, SOLVERS
    models as models,  # noqa: F401  -> MODELS
)
from cyfm.core import (
    COUPLINGS,
    DATASETS,
    MANIFOLDS,
    MODELS,
    SOLVERS,
    BaseComplexDataset,
    BaseManifold,
    BaseODESolver,
    BaseSDESolver,
    Coupling,
    Registry,
    Sampler,
    VelocityField,
)
from cyfm.core.manifold import Representation
from cyfm.manifolds.cylindrical import CylindricalManifold
from cyfm.manifolds.euclidean import EuclideanManifold

try:
    __version__ = version("cyfm")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0.dev0"

__all__ = [
    "COUPLINGS",
    "DATASETS",
    "MANIFOLDS",
    "MODELS",
    "SOLVERS",
    "BaseComplexDataset",
    "BaseManifold",
    "BaseODESolver",
    "BaseSDESolver",
    "Coupling",
    "CylindricalManifold",
    "EuclideanManifold",
    "Registry",
    "Representation",
    "Sampler",
    "VelocityField",
    "__version__",
    "core",
]

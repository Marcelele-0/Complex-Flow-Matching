"""Complex Flow Matching (CFM) package."""

from cyfm import core
from cyfm.core import (
    DATASETS,
    LOSSES,
    MANIFOLDS,
    MODELS,
    SOLVERS,
    BaseComplexDataset,
    BaseLoss,
    BaseManifold,
    BaseODESolver,
    CylindricalLoss,
    EuclideanLoss,
    Registry,
)
from cyfm.manifolds.cylindrical import CylindricalManifold
from cyfm.manifolds.euclidean import EuclideanManifold

__all__ = [
    "core",
    "Registry",
    "MANIFOLDS",
    "MODELS",
    "SOLVERS",
    "LOSSES",
    "DATASETS",
    "BaseManifold",
    "BaseODESolver",
    "BaseLoss",
    "CylindricalLoss",
    "EuclideanLoss",
    "CylindricalManifold",
    "EuclideanManifold",
    "BaseComplexDataset",
]

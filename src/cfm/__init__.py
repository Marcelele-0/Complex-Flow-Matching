"""Complex Flow Matching (CFM) package."""

from cfm import core
from cfm.core import (
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
from cfm.manifolds.cylindrical import CylindricalManifold
from cfm.manifolds.euclidean import EuclideanManifold

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

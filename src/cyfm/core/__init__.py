"""Core modular architecture and registries for Complex Flow Matching (CFM)."""

from __future__ import annotations

from cyfm.core.dataset import BaseComplexDataset
from cyfm.core.loss import BaseLoss, CylindricalLoss, EuclideanLoss
from cyfm.core.manifold import BaseManifold
from cyfm.core.registry import (
    DATASETS,
    LOSSES,
    MANIFOLDS,
    MASKS,
    MODELS,
    SOLVERS,
    Registry,
)
from cyfm.core.solver import BaseODESolver

__all__ = [
    "Registry",
    "MANIFOLDS",
    "MODELS",
    "SOLVERS",
    "LOSSES",
    "DATASETS",
    "MASKS",
    "BaseManifold",
    "BaseODESolver",
    "BaseLoss",
    "CylindricalLoss",
    "EuclideanLoss",
    "BaseComplexDataset",
]

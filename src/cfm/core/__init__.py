"""Core modular architecture and registries for Complex Flow Matching (CFM)."""

from __future__ import annotations

from cfm.core.dataset import BaseComplexDataset
from cfm.core.loss import BaseLoss, CylindricalLoss, EuclideanLoss
from cfm.core.manifold import BaseManifold
from cfm.core.registry import (
    DATASETS,
    LOSSES,
    MANIFOLDS,
    MASKS,
    MODELS,
    SOLVERS,
    Registry,
)
from cfm.core.solver import BaseODESolver

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

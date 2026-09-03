"""Core modular architecture and registries for Complex Flow Matching (CFM)."""

from __future__ import annotations

from cfm.core.dataset import BaseComplexDataset
from cfm.core.loss import BaseLoss, CylindricalLoss, EuclideanLoss
from cfm.core.manifold import BaseManifold
from cfm.core.reconstructor import (
    BaseReconstructor,
    FlowMatchingReconstructor,
    ZeroFilledReconstructor,
)
from cfm.core.registry import (
    DATASETS,
    LOSSES,
    MANIFOLDS,
    MODELS,
    RECONSTRUCTORS,
    SOLVERS,
    Registry,
)
from cfm.core.solver import BaseODESolver, BaseSDESolver

__all__ = [
    "Registry",
    "MANIFOLDS",
    "MODELS",
    "SOLVERS",
    "LOSSES",
    "DATASETS",
    "RECONSTRUCTORS",
    "BaseManifold",
    "BaseODESolver",
    "BaseSDESolver",
    "BaseLoss",
    "CylindricalLoss",
    "EuclideanLoss",
    "BaseReconstructor",
    "ZeroFilledReconstructor",
    "FlowMatchingReconstructor",
    "BaseComplexDataset",
]

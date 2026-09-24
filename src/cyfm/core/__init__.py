"""Core modular architecture and registries for CyFM."""

from __future__ import annotations

from cyfm.core.dataset import BaseComplexDataset
from cyfm.core.manifold import BaseManifold
from cyfm.core.protocols import Coupling, Sampler, VelocityField
from cyfm.core.registry import (
    COUPLINGS,
    DATASETS,
    MANIFOLDS,
    MODELS,
    SOLVERS,
    Registry,
)
from cyfm.core.solver import BaseODESolver, BaseSDESolver

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
    "Registry",
    "Sampler",
    "VelocityField",
]

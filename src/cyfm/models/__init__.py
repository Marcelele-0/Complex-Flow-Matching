"""Model architectures for CyFM."""

from __future__ import annotations

from cyfm.models.mlp import PointwiseVelocityMLP
from cyfm.models.unet import CylindricalUNet

__all__ = [
    "CylindricalUNet",
    "PointwiseVelocityMLP",
]

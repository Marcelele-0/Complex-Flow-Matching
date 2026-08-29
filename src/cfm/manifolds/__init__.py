"""Geometry registry: the one place a manifold name is resolved to an implementation.

Register a new geometry in :func:`build_manifold` and every entry point picks it
up, the same way :func:`cfm.utils.inference.build_model` centralises architectures.
"""

from __future__ import annotations

from omegaconf import DictConfig

from cfm.manifolds.base import Manifold
from cfm.manifolds.cylindrical import CylindricalManifold
from cfm.manifolds.euclidean import EuclideanManifold

__all__ = [
    "CylindricalManifold",
    "EuclideanManifold",
    "Manifold",
    "build_manifold",
]


def build_manifold(cfg: DictConfig) -> Manifold:
    """Instantiate the geometry named by ``cfg.manifold.name``.

    Loss hyper-parameters are read from ``cfg.training.loss`` rather than from
    the manifold group, so the existing overrides (``training.loss.lambda_phase``,
    ``training.loss.lambda_hf``, ...) and ``schedule_runs.sh`` keep working
    unchanged, and one training config serves both geometries.

    Args:
        cfg: Full Hydra config. Reads ``manifold.name``, ``manifold.noise_prior``
            (Euclidean only) and the relevant keys under ``training.loss``.

    Returns:
        The configured manifold. Defaults to cylindrical when nothing is set, so
        a config predating the manifold group behaves as it always did.

    Raises:
        ValueError: If ``cfg.manifold.name`` is not a known geometry.
    """
    manifold_cfg = cfg.get("manifold", {}) or {}
    name = manifold_cfg.get("name", "cylindrical")
    loss_cfg = cfg.get("training", {}).get("loss", {}) or {}

    # Shared by both geometries; see cfm.flow.spectral.
    lambda_hf = loss_cfg.get("lambda_hf", 0.0)
    hf_boost_factor = loss_cfg.get("hf_boost_factor", 4.0)

    match name:
        case "cylindrical":
            manifold: Manifold = CylindricalManifold(
                amp_loss_type=loss_cfg.get("amp_loss_type", "l1"),
                phase_loss_type=loss_cfg.get("phase_loss_type", "l1"),
                lambda_phase=loss_cfg.get("lambda_phase", 1.0),
                lambda_hf=lambda_hf,
                hf_boost_factor=hf_boost_factor,
            )

        case "euclidean":
            manifold = EuclideanManifold(
                noise_prior=manifold_cfg.get("noise_prior", "uniform"),
                loss_type=loss_cfg.get("vel_loss_type", "l1"),
                lambda_hf=lambda_hf,
                hf_boost_factor=hf_boost_factor,
            )

        case _:
            raise ValueError(
                f"Unknown manifold name specified in config: {name}. "
                f"Expected one of ['cylindrical', 'euclidean']."
            )

    print(f"Using manifold: {manifold.name} ({manifold.state_channels}-channel state)")
    return manifold

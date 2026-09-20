"""Geometry registry: the one place a manifold name is resolved to an implementation.

Register a new geometry in :func:`build_manifold` and every entry point picks it
up, the same way :func:`cyfm.utils.inference.build_model` centralises architectures.
"""

from __future__ import annotations

from omegaconf import DictConfig

from cyfm.core.manifold import BaseManifold
from cyfm.core.registry import MANIFOLDS
from cyfm.manifolds.base import Manifold
from cyfm.manifolds.complex_diffusion import ComplexDiffusionManifold
from cyfm.manifolds.cylindrical import CylindricalManifold
from cyfm.manifolds.euclidean import EuclideanManifold

__all__ = [
    "BaseManifold",
    "ComplexDiffusionManifold",
    "CylindricalManifold",
    "EuclideanManifold",
    "Manifold",
    "build_manifold",
]


def build_manifold(cfg: DictConfig) -> BaseManifold:
    """Instantiate the geometry named by ``cfg.manifold.name``.

    Uses the MANIFOLDS registry to dynamically resolve and build the manifold instance.

    Loss hyper-parameters are read from ``cfg.training.loss`` rather than from
    the manifold group, so the existing overrides (``training.loss.lambda_phase``,
    ``training.loss.amp_loss_type``, ...) keep working unchanged, and one training
    config serves both geometries.

    Args:
        cfg: Full Hydra config. Reads ``manifold.name``, ``manifold.noise_prior``
            (Euclidean only) and the relevant keys under ``training.loss``.

    Returns:
        The configured manifold. Defaults to cylindrical when nothing is set, so
        a config predating the manifold group behaves as it always did.

    Raises:
        ValueError: If ``cfg.manifold.name`` is not a known geometry in MANIFOLDS.
    """
    manifold_cfg = cfg.get("manifold", {}) or {}
    name = manifold_cfg.get("name", "cylindrical")
    loss_cfg = cfg.get("training", {}).get("loss", {}) or {}

    if not MANIFOLDS.contains(name):
        raise ValueError(
            f"Unknown manifold name specified in config: {name}. "
            f"Expected one of {MANIFOLDS.list()}."
        )

    match name:
        case "cylindrical":
            manifold: BaseManifold = MANIFOLDS.build(
                "cylindrical",
                amp_loss_type=loss_cfg.get("amp_loss_type", "l1"),
                phase_loss_type=loss_cfg.get("phase_loss_type", "l1"),
                lambda_phase=loss_cfg.get("lambda_phase", 1.0),
                phase_weight=manifold_cfg.get("phase_weight", 1.0),
                phase_spread=manifold_cfg.get("phase_spread", None),
                spatial_correlation=manifold_cfg.get("spatial_correlation", None),
                phase_amplitude_weighting=loss_cfg.get("phase_amplitude_weighting", True),
            )

        case "euclidean":
            manifold = MANIFOLDS.build(
                "euclidean",
                noise_prior=manifold_cfg.get("noise_prior", "uniform"),
                spatial_correlation=manifold_cfg.get("spatial_correlation", None),
                loss_type=loss_cfg.get("vel_loss_type", "l1"),
            )

        case "complex_diffusion":
            # Only the keys the config actually carries are forwarded, so the
            # constructor stays the single source of truth for every default. The
            # branch above re-types its defaults and that is how they drift: this
            # arm's sigma_max in particular is calibrated to the data, and a second
            # default here would silently win whenever a config omits it.
            # The loss group is not read: score matching has no velocity loss
            # type to configure.
            settings = {
                key: manifold_cfg[key]
                for key in (
                    "sigma_min",
                    "sigma_max",
                    "eps",
                    "likelihood_weighting",
                    "snr",
                    "corrector_steps",
                    "num_steps",
                    "nfe_mode",
                )
                if key in manifold_cfg
            }
            manifold = MANIFOLDS.build("complex_diffusion", **settings)

        case _:
            manifold = MANIFOLDS.build(name, **manifold_cfg)

    print(f"Using manifold: {manifold.name} ({manifold.state_channels}-channel state)")
    return manifold

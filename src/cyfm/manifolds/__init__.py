"""Geometry registry: the one place a manifold name is resolved to an implementation.

Register a new geometry with ``@register_manifold`` and every entry point picks
it up. There is deliberately no branch here per geometry: each class reads its
own config through :meth:`~cyfm.core.manifold.BaseManifold.from_config`, so
adding one means writing the class and nothing else.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cyfm.config.resolve import as_plain_dict
from cyfm.core.manifold import BaseManifold
from cyfm.core.registry import MANIFOLDS
from cyfm.manifolds.complex_diffusion import ComplexDiffusionManifold
from cyfm.manifolds.cylindrical import CylindricalManifold
from cyfm.manifolds.euclidean import EuclideanManifold

DEFAULT_MANIFOLD = "cylindrical"

__all__ = [
    "DEFAULT_MANIFOLD",
    "BaseManifold",
    "ComplexDiffusionManifold",
    "CylindricalManifold",
    "EuclideanManifold",
    "build_manifold",
]


def build_manifold(
    manifold_cfg: Mapping[str, Any] | None,
    loss_cfg: Mapping[str, Any] | None = None,
) -> BaseManifold:
    """Instantiate the geometry named by ``manifold_cfg["name"]``.

    Loss hyper-parameters come from the ``training.loss`` group rather than the
    manifold group, so the existing overrides (``training.loss.lambda_phase``,
    ``training.loss.amp_loss_type``, ...) keep working unchanged and one training
    config serves both geometries. Which of those keys a geometry reads is
    declared on the geometry, as
    :attr:`~cyfm.core.manifold.BaseManifold.config_loss_keys`.

    Args:
        manifold_cfg: The ``manifold`` config group, as a plain mapping or a
            ``DictConfig``. ``name`` picks the registry key.
        loss_cfg: The ``training.loss`` group, or ``None``.

    Returns:
        The configured manifold. Defaults to cylindrical when nothing is set, so
        a config predating the manifold group behaves as it always did.

    Raises:
        ValueError: If the name is not a known geometry, or the group carries a
            key that geometry cannot accept.
    """
    manifold = as_plain_dict(manifold_cfg)
    loss = as_plain_dict(loss_cfg)

    name = str(manifold.get("name", DEFAULT_MANIFOLD))
    if not MANIFOLDS.contains(name):
        raise ValueError(
            f"Unknown manifold name specified in config: {name}. "
            f"Expected one of {MANIFOLDS.list()}."
        )

    # Deliberately silent. This used to print "Using manifold: ...", which made a
    # script's output depend on how many manifolds it happened to construct and
    # put stray lines in the reproduction reports. The entry points announce the
    # geometry themselves, where the announcement belongs.
    return MANIFOLDS.get_class(name).from_config(manifold, loss)

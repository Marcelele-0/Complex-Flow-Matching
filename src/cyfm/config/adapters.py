"""Full-config adapters: the seam between a composed Hydra tree and the library.

The builders in the library proper take the config *group* they need --
``build_manifold(manifold_cfg, loss_cfg)``, ``build_dataset(dataset_cfg)`` -- and
plain mappings at that, so a script or a test can call one with a dict literal
and no Hydra anywhere. The entry points hold a whole composed tree and would
otherwise each repeat the same two lines of key-plucking; the functions here do
it once.

Nothing below is a second way to build anything. Each is the extraction and
nothing more, so a caller that already has the groups should skip it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cyfm.core.manifold import BaseManifold
from cyfm.manifolds import build_manifold

__all__ = ["manifold_from_config"]


def manifold_from_config(cfg: Mapping[str, Any]) -> BaseManifold:
    """Build the geometry from a composed config tree.

    Args:
        cfg: The whole config. Reads the ``manifold`` group and
            ``training.loss``, which is where the loss hyper-parameters live so
            that one training config serves both geometries.

    Returns:
        The configured manifold.
    """
    training = cfg.get("training") or {}
    return build_manifold(cfg.get("manifold"), training.get("loss"))

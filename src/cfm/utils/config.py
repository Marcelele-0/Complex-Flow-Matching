"""Helpers for turning Hydra config nodes into plain Python containers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

__all__ = ["as_plain_dict"]


def as_plain_dict(node: Any) -> dict[str, Any]:
    """Resolve a Hydra config node to a plain ``dict``.

    The dataset constructors take mask specifications as ordinary mappings so they
    stay usable from tests and scripts without Hydra in the picture. ``None`` and
    missing nodes collapse to an empty dict so callers can merge unconditionally.

    Args:
        node: A ``DictConfig``, a mapping, or ``None``.

    Returns:
        A plain dictionary with interpolations resolved.
    """
    if node is None:
        return {}
    if isinstance(node, DictConfig):
        return dict(cast(dict[str, Any], OmegaConf.to_container(node, resolve=True)))
    if isinstance(node, Mapping):
        return dict(node)
    raise TypeError(f"Expected a mapping or DictConfig, got {type(node).__name__}")

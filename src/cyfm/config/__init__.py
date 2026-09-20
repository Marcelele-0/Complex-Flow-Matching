"""The Hydra boundary.

Everything in this package knows about ``DictConfig``; nothing outside it should.
The library proper -- geometries, bridges, couplings, solvers, metrics -- takes
plain Python arguments, so it can be imported and used without Hydra in the
picture, while the reproduction layer keeps composing the configs the archives
record.
"""

from __future__ import annotations

from cyfm.config.adapters import manifold_from_config
from cyfm.config.resolve import as_plain_dict

__all__ = ["as_plain_dict", "manifold_from_config"]

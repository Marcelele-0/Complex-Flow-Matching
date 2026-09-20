"""The interface every geometry implements, and the only thing the entry points know.

``train.py`` and ``evaluate.py`` are written against
:class:`Manifold` and nothing else. Everything that is *not* a member below -
the dataset, the U-Net trunk, the optimizer, the Heun schedule, the metrics, the
accumulator, the checkpoint plumbing - is therefore shared by construction and
cannot drift between the two arms of the comparison. That is the property the
side-by-side experiment depends on: any difference in the reported numbers has
to come from one of the methods here.
"""

from __future__ import annotations

from cyfm.core.manifold import BaseManifold

# Retain Manifold alias for 100% backward compatibility
Manifold = BaseManifold

__all__ = ["Manifold", "BaseManifold"]

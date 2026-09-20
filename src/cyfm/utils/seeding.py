"""One place where a run's randomness is fixed.

Seeding was inline in ``train.py`` and again, differently, in ``evaluate.py``:
one seeded torch globally and built a loader generator, the other built two
generators (one on the device for the prior draw, one on the CPU for the metrics)
and left the global alone. Both are right for what they do; having them written
out at the two call sites is what let them drift.
"""

from __future__ import annotations

import torch

__all__ = ["seed_everything"]


def seed_everything(seed: int | None) -> torch.Generator | None:
    """Seed torch globally and return a matching dataloader generator.

    One seed puts both geometries on element-wise identical weights everywhere
    except ``init_conv`` -- the single module whose shape depends on the geometry,
    built last so it cannot shift the RNG for anything else. Initialisation is
    therefore not a confound in the side-by-side comparison. Data order, the noise
    draw and the time sample still vary with the seed, so several seeds are run
    per arm and the spread is reported.

    Args:
        seed: The seed, or ``None`` for an unseeded run.

    Returns:
        A CPU generator seeded the same way, for a ``DataLoader`` to shuffle
        with, or ``None`` when ``seed`` is ``None``.
    """
    if seed is None:
        return None
    torch.manual_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator

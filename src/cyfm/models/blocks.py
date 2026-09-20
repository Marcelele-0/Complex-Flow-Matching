"""Building blocks shared by more than one architecture.

Only what is genuinely shared lives here. :class:`SinusoidalPositionEmbeddings`
was implemented twice, once in the U-Net and once in the MLP; the two agreed
bitwise for every embedding width above 2, and disagreed only in that the
U-Net's copy divided by ``dim // 2 - 1`` without a guard and so raised
ZeroDivisionError at ``dim == 2``. The surviving version is the guarded one,
which also validates its width.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    """Standard sinusoidal embedding of a scalar time.

    Args:
        dim: Embedding width, must be positive and even.

    Raises:
        ValueError: If ``dim`` is not a positive even number.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(f"embedding dim must be positive and even, got {dim}")
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Embed a batch of times.

        Args:
            time: Times of shape ``[B]``.

        Returns:
            Embedding of shape ``[B, dim]``.
        """
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        frequencies = torch.exp(torch.arange(half, device=time.device) * -scale)
        angles = time.reshape(-1, 1) * frequencies.reshape(1, -1)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

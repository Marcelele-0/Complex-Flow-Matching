"""A pointwise velocity field: the same MLP applied at every coefficient.

The U-Nets in this package earn their keep on data with spatial structure. On a
field whose coefficients are drawn independently there is no structure to
exploit, and a U-Net there is an expensive way to be a multilayer perceptron.
This model is the honest baseline for that case, and it is the control that makes
the structured case interpretable: if a U-Net does no better than this on a field
that *does* have structure, it is not using the structure.

It is also what makes the synthetic gates reachable from the ordinary entry
points. With ``shared_encoding: true`` it reproduces the velocity field the gate
scripts train, including the confound control the gates depend on -- both
geometries receive the identical encoding ``(re, im, m, cos phi, sin phi)``, so a
difference between them cannot be a difference in what the network was allowed to
see. The default is ``false``, which follows the rest of the package: each
geometry hands the trunk its own state, ``(m, cos phi, sin phi)`` or ``(re, im)``.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from cyfm.core.registry import MODELS

# Below this amplitude a complex number's phase is numerically meaningless, so
# the shared encoding reports a fixed direction rather than atan2 noise. The
# cylinder chart is genuinely singular at the origin; this is where that shows.
_AMPLITUDE_FLOOR = 1e-8

# Width of the shared encoding: (re, im, m, cos phi, sin phi).
_SHARED_ENCODING_WIDTH = 5


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


@MODELS.register("mlp")
@MODELS.register("pointwise_mlp")
class PointwiseVelocityMLP(nn.Module):
    """Velocity field applied independently at every spatial coefficient.

    Args:
        in_channels: Width of the state, normally ``Manifold.state_channels``:
            3 for the cylinder ``(m, cos phi, sin phi)``, 2 for the plane
            ``(re, im)``. Ignored for the trunk's input width when
            ``shared_encoding`` is set, but still used to decode the state.
        out_channels: Width of the emitted velocity, 2 for both geometries.
        base_channels: Hidden width.
        depth: Number of hidden layers, at least one.
        shared_encoding: Feed both geometries the identical five-channel
            encoding instead of their own state. This is the gates' confound
            control; leave it off to match the rest of the package.
        time_embedding_dim: Width of the sinusoidal time embedding. ``0``
            appends the raw scalar time instead, which is what the gate scripts
            do.

    Raises:
        ValueError: If any width or depth is out of range, or ``in_channels`` is
            not 2 or 3 while ``shared_encoding`` is set.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        base_channels: int = 128,
        depth: int = 3,
        shared_encoding: bool = False,
        time_embedding_dim: int = 0,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels < 1:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if base_channels < 1:
            raise ValueError(f"base_channels must be positive, got {base_channels}")
        if depth < 1:
            raise ValueError(f"depth must be positive, got {depth}")
        if shared_encoding and in_channels not in (2, 3):
            raise ValueError(
                f"shared_encoding needs a plane (2) or cylinder (3) state to decode, "
                f"got in_channels={in_channels}"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shared_encoding = shared_encoding

        self.time_embedding = (
            SinusoidalPositionEmbeddings(time_embedding_dim) if time_embedding_dim else None
        )
        time_width = time_embedding_dim if time_embedding_dim else 1
        state_width = _SHARED_ENCODING_WIDTH if shared_encoding else in_channels

        layers: list[nn.Module] = [nn.Linear(state_width + time_width, base_channels), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(base_channels, base_channels), nn.SiLU()])
        layers.append(nn.Linear(base_channels, out_channels))
        self.net = nn.Sequential(*layers)

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """Turn a state into the trunk's input features.

        Args:
            state: State of shape ``[N, in_channels]``, already flattened over
                batch and space.

        Returns:
            Features of shape ``[N, state_width]``.
        """
        if not self.shared_encoding:
            return state
        if self.in_channels == 3:
            amplitude = state[:, 0].clamp_min(0.0)
            cosine, sine = state[:, 1], state[:, 2]
            real, imag = amplitude * cosine, amplitude * sine
        else:
            real, imag = state[:, 0], state[:, 1]
            amplitude = torch.sqrt(real * real + imag * imag).clamp_min(_AMPLITUDE_FLOOR)
            cosine, sine = real / amplitude, imag / amplitude
        return torch.stack([real, imag, amplitude, cosine, sine], dim=1)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Predict the tangent velocity at every coefficient.

        Args:
            x: State of shape ``[B, in_channels, H, W]``.
            time: Times of shape ``[B]``, or any shape whose first dimension
                broadcasts against the batch.

        Returns:
            Velocity of shape ``[B, out_channels, H, W]``.

        Raises:
            ValueError: If ``x`` is not 4D with ``in_channels`` channels.
        """
        if x.ndim != 4:
            raise ValueError(f"expected [B, C, H, W], got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {x.shape[1]}")

        batch, _, height, width = x.shape
        flat = x.permute(0, 2, 3, 1).reshape(-1, self.in_channels)
        features = self.encode(flat)

        scalar_time = time.reshape(batch, -1)[:, 0].to(x.dtype)
        embedded = (
            self.time_embedding(scalar_time)
            if self.time_embedding is not None
            else scalar_time.reshape(batch, 1)
        )
        # One time per sample, repeated across that sample's coefficients.
        per_coefficient = embedded.repeat_interleave(height * width, dim=0)

        velocity = self.net(torch.cat([features, per_coefficient], dim=1))
        return velocity.reshape(batch, height, width, self.out_channels).permute(0, 3, 1, 2)

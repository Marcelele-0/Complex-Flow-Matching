"""The paper's U-Net, and the blocks it is built from.

One trunk serves both geometries. The only architectural difference between the
cylindrical and Cartesian arms is the width of the first convolution, which
follows ``Manifold.state_channels``: three for ``(m, cos phi, sin phi)``, two for
``(Re z, Im z)``. Everything after it is identical, which is what makes a
measured difference between the arms a difference in geometry.

The trunk is a symmetric encoder/decoder with skip connections, ``num_levels``
downsampling stages and self-attention at the coarsest resolution. The defaults
(``base_channels=64``, ``num_levels=3``, ``use_attention=True``) are the shipped
architecture; both are constructor arguments so the depth and the attention can
be dialled from config without a second class. ``num_levels`` downsampling stages
require the field's spatial size to be divisible by ``2 ** num_levels``;
``CROP_BASE`` in the training pipeline already enforces divisibility by 16, which
covers up to four levels.
"""

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from cyfm.core.registry import register_model
from cyfm.models.blocks import SinusoidalPositionEmbeddings


class TimeConditionedBlock(nn.Module):
    """Residual convolutional block conditioned on diffusion time embedding.

    Args:
        in_channels: Input feature channels.
        out_channels: Output feature channels.
        time_emb_dim: Dimension of time embedding vector.
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.silu = nn.SiLU()

        # Residual connection aligning channels if there was a change
        self.residual = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Convolve, inject the time embedding, convolve again, add the residual."""
        # Main convolutional branch
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.silu(h)

        # Time injection (broadcasting over H and W dimensions)
        time_info = self.time_mlp(t_emb)
        time_info = time_info.unsqueeze(-1).unsqueeze(-1)
        h = h + time_info

        h = self.conv2(h)
        h = self.norm2(h)

        return self.silu(h + self.residual(x))


class AttentionBlock(nn.Module):
    """Multi-head self-attention over spatial positions, with a residual add.

    Placed at the coarsest resolution of the U-Net, where the token count
    ``H * W`` is smallest: the convolutional trunk mixes only local
    neighbourhoods, so a field whose global layout matters -- a knee at 320x320,
    whose bottleneck is 40x40 here -- has no path for one region to see another
    without this. The attention is *not* time-conditioned; the
    :class:`TimeConditionedBlock` s on either side of it carry the time signal.

    The score matrix is never materialised: ``scaled_dot_product_attention``
    fuses the softmax, so the memory cost is linear rather than quadratic in the
    token count, which is what keeps the 320x320 bottleneck affordable.

    Args:
        channels: Feature channels in and out. Must be divisible by both 8 (its
            :class:`~torch.nn.GroupNorm`) and ``num_heads``.
        num_heads: Number of attention heads.

    Raises:
        ValueError: If ``channels`` is not divisible by ``num_heads``.
    """

    def __init__(self, channels: int, num_heads: int = 8) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"attention channels ({channels}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend over every spatial position and add the result back to ``x``.

        Args:
            x: Feature map ``[B, C, H, W]``.

        Returns:
            Feature map ``[B, C, H, W]``.
        """
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        # [B, 3C, H, W] -> three [B, heads, H*W, head_dim] tensors.
        q, k, v = qkv.chunk(3, dim=1)

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(b, self.num_heads, c // self.num_heads, h * w).transpose(-2, -1)

        attended = F.scaled_dot_product_attention(heads(q), heads(k), heads(v))
        attended = attended.transpose(-2, -1).reshape(b, c, h, w)
        return x + self.proj(attended)


@register_model("c_unet")
@register_model("cylindrical_unet")
class CylindricalUNet(nn.Module):
    """U-Net for Continuous Flow Matching on complex MRI representations.

    A symmetric encoder/decoder with skip connections. Each encoder stage doubles
    the channel width and halves the resolution; the decoder mirrors it, and
    self-attention runs at the bottleneck. The widths are ``base_channels * 2**i``,
    so ``base_channels`` divisible by 8 keeps every :class:`~torch.nn.GroupNorm`
    and the attention heads well defined.

    Args:
        base_channels: Width of the first feature level. Must be divisible by 8.
        in_channels: Channels of input state tensor [B, C_in, H, W].
        out_channels: Channels of output velocity field [B, C_out, H, W].
        num_levels: Number of downsampling stages. The input's spatial size must
            be divisible by ``2 ** num_levels``.
        use_attention: Whether to run self-attention at the bottleneck.
        velocity_bound: Optional per-channel ceiling on ``|velocity|``.

    Raises:
        ValueError: If ``base_channels`` is not a positive multiple of 8, or
            ``num_levels`` is not positive, or ``velocity_bound`` is the wrong
            width or non-positive.
    """

    def __init__(
        self,
        base_channels: int = 64,
        in_channels: int = 3,
        out_channels: int = 2,
        num_levels: int = 3,
        use_attention: bool = True,
        velocity_bound: Sequence[float] | None = None,
    ) -> None:
        super().__init__()

        # Divisible by 8 for every GroupNorm(8, base*2**i), and so the attention
        # heads below divide the bottleneck width. A cryptic GroupNorm error at
        # construction is replaced by a statement of the actual requirement.
        if base_channels <= 0 or base_channels % 8 != 0:
            raise ValueError(f"base_channels must be a positive multiple of 8, got {base_channels}")
        if num_levels <= 0:
            raise ValueError(f"num_levels must be positive, got {num_levels}")
        self.num_levels = num_levels

        # Per-channel ceiling on |velocity|, applied as bound * tanh(logit / bound) so the
        # map is the identity to first order and saturates only where the target cannot
        # go. Each geometry supplies its own from its own metric, so this is not an
        # advantage handed to one arm: on the cylinder the angular coordinate lives on a
        # circle of diameter pi and the bound binds; in the plane the bound is the support
        # diameter and is slack. The marginal field a flow-matching model converges to is
        # a conditional expectation of targets that already respect these limits, so the
        # optimum lies strictly inside and nothing representable is lost.
        if velocity_bound is None:
            self.register_buffer("_bound", None)
        else:
            bound = torch.tensor(list(velocity_bound), dtype=torch.float32)
            if bound.numel() != out_channels:
                raise ValueError(
                    f"velocity_bound has {bound.numel()} entries for {out_channels} channels"
                )
            if not bool((bound > 0).all()):
                raise ValueError(f"velocity_bound must be positive, got {velocity_bound}")
            self.register_buffer("_bound", bound.reshape(1, -1, 1, 1))

        time_emb_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # ENCODER: level i widens base*2**i -> base*2**(i+1) before a 2x pool, so
        # each skip is taken at the pre-pool width and resolution.
        self.down_blocks = nn.ModuleList(
            TimeConditionedBlock(
                base_channels * 2**i, base_channels * 2 ** (i + 1), time_emb_dim
            )
            for i in range(num_levels)
        )
        # Parameter-free, so one instance is reused at every stage.
        self.pool = nn.MaxPool2d(2)

        # BOTTLENECK at base*2**num_levels, with self-attention over the coarsest grid.
        bottleneck_channels = base_channels * 2**num_levels
        self.bottleneck = TimeConditionedBlock(
            bottleneck_channels, bottleneck_channels, time_emb_dim
        )
        self.bottleneck_attn = AttentionBlock(bottleneck_channels) if use_attention else None

        # DECODER: mirror the encoder, deepest first. Each stage upsamples the
        # running feature (base*2**(i+1)) and concatenates the matching skip of
        # the same width, so the block consumes base*2**(i+2) and emits base*2**i.
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up_blocks = nn.ModuleList(
            TimeConditionedBlock(
                base_channels * 2 ** (i + 2), base_channels * 2**i, time_emb_dim
            )
            for i in reversed(range(num_levels))
        )

        # Final convolution outputting the velocity field.
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        # Initial convolution, from the manifold's state channels. This is the one
        # module whose shape depends on the manifold, so constructing it last means
        # every other module has already drawn from the RNG at the same stream
        # position in both geometries: with one seed, the cylindrical and Euclidean
        # models start from element-wise identical weights everywhere except this
        # convolution. Moving it earlier silently reintroduces an initialisation
        # confound. Pinned by tests/test_manifolds/test_manifolds.py.
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Predict velocity field for state at given diffusion time.

        Args:
            x: Manifold state tensor [B, in_channels, H, W]. ``H`` and ``W`` must
                be divisible by ``2 ** num_levels``.
            time: Diffusion time values [B].

        Returns:
            Predicted velocity field [B, out_channels, H, W].

        Raises:
            ValueError: If ``H`` or ``W`` is not divisible by ``2 ** num_levels``,
                which would make a decoder skip concatenation shape-mismatch.
        """
        stride = 2**self.num_levels
        height, width = x.shape[-2], x.shape[-1]
        if height % stride or width % stride:
            raise ValueError(
                f"input {height}x{width} is not divisible by 2**num_levels={stride}; "
                "the decoder skip connections would not line up"
            )

        t_emb = self.time_mlp(time)

        x = self.init_conv(x)

        # Encoder, keeping the pre-pool activation of each level as its skip.
        skips: list[torch.Tensor] = []
        for block in self.down_blocks:
            x = block(x, t_emb)
            skips.append(x)
            x = self.pool(x)

        # Bottleneck, then global self-attention over the coarsest grid.
        x = self.bottleneck(x, t_emb)
        if self.bottleneck_attn is not None:
            x = self.bottleneck_attn(x)

        # Decoder: up_blocks are ordered deepest-first, matching the reversed skips.
        for block, skip in zip(self.up_blocks, reversed(skips), strict=True):
            x = self.up(x)
            x = torch.cat([x, skip], dim=1)
            x = block(x, t_emb)

        velocity = self.final_conv(x)
        bound = self._bound
        if bound is None:
            return velocity
        assert isinstance(bound, torch.Tensor)
        return bound * torch.tanh(velocity / bound)

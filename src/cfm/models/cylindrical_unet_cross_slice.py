"""2.5D cylindrical U-Net: shared 2D encoders per slice, fused by cross-slice attention.

A full 3D U-Net over an MRI volume is expensive and mostly wasteful: neighbouring
slices are highly correlated, so the useful cross-slice signal is a thin
correction rather than a whole extra dimension of convolution. This model keeps
the encoder and decoder purely 2D and buys volumetric consistency with a single
attention step at the bottleneck.

Data flow, for a window of ``S`` neighbouring slices centred on ``z``::

    h_s      = Shared2DEncoder(x_s, t)        for every s in {z-S//2 .. z+S//2}
    H'       = CrossSliceAttention(stack(h_s))          # bottleneck only
    v_center = Shared2DDecoder(H'_center, skips_center, t)

Only the **center** slice's velocity is produced, so the loss supervises the
center alone. Neighbours exist to inform it, not to be predicted. That is why
this model cannot yet drive :class:`~cfm.flow.solver.CylindricalODESolver`, which
needs a velocity matching the shape of the state it is advancing.

Weight sharing is structural rather than enforced: the encoder is applied once to
a tensor with the slice axis folded into the batch, so there are no per-slice
parameters and ``S`` may change between calls without touching the model.

Memory versus :class:`~cfm.models.cylindrical_unet_attention.CylindricalUNetAttention`
--------------------------------------------------------------------------------------
Two effects pull in opposite directions and the net result is not obvious a priori:

* The encoder runs on ``B*S`` images instead of ``B``, so encoder activations
  scale roughly with ``S``. The decoder is unaffected: it runs at ``B``.
* The bottleneck gets cheaper. Spatial self-attention in the baseline costs
  ``O((h*w)^2)`` per sample, which is 1024 tokens attending to 1024 at a 32x32
  bottleneck. Cross-slice attention costs ``O(S^2)`` per spatial position, i.e.
  9 for ``S=3``.

Measure with ``torch.cuda.max_memory_allocated()`` before trusting any figure;
as a starting point, expect to roughly halve ``batch_size`` at ``S=3``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Reuse the building blocks rather than adding a third copy to the codebase.
from cfm.models.cylindrical_unet_attention import (
    SinusoidalPositionEmbeddings,
    TimeConditionedBlock,
)


class CrossSliceAttention(nn.Module):
    """Multi-head attention along the slice axis, independently per spatial position.

    Each spatial position of the bottleneck feature map becomes its own sequence
    of ``S`` tokens, one per slice, so a pixel in the center slice can look at the
    same anatomical location in its neighbours. Spatial mixing is deliberately not
    done here: the 2D encoder already handles that.

    The sequence length is ``S`` (typically 3), which makes this far cheaper than
    the spatial self-attention it replaces.
    """

    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse information across slices.

        Args:
            x: Bottleneck features, shape ``[B, S, C, h, w]``.

        Returns:
            Fused features of the same shape, as a residual update of ``x``.
        """
        b, s, c, h, w = x.shape

        # GroupNorm expects [N, C, H, W], so normalise in the folded layout.
        normed = self.norm(x.reshape(b * s, c, h, w)).reshape(b, s, c, h * w)
        # Sequence of spatial positions across slices
        seq = normed.permute(0, 3, 1, 2).reshape(b * h * w, s, c)

        attn_out, _ = self.mha(seq, seq, seq, need_weights=False)

        # Reshape back to B x S x C x H x W
        attn_out = attn_out.reshape(b, h * w, s, c).permute(0, 2, 3, 1).reshape(b, s, c, h, w)
        return x + attn_out


class CylindricalUNetCrossSlice(nn.Module):
    """2.5D U-Net predicting the center slice's velocity from a window of slices.

    Input is ``[B, S, 3, H, W]`` (amplitude, cos(phi), sin(phi) per slice) and
    output is ``[B, 2, H, W]`` (v_amp, v_phi) for the center slice only.

    Args:
        base_channels: Width of the first encoder stage.
        channel_mults: Per-stage width multipliers. Length minus one is the number
            of pooling levels, so ``H`` and ``W`` must be divisible by
            ``2 ** (len(channel_mults) - 1)``.
        attn_heads: Heads for the cross-slice attention.
    """

    def __init__(
        self,
        base_channels: int = 96,
        channel_mults: list | None = None,
        attn_heads: int = 4,
    ):
        super().__init__()

        if channel_mults is None:
            channel_mults = [1, 2, 4, 8, 8]
        self.channel_mults = channel_mults
        time_emb_dim = base_channels * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(3, base_channels, kernel_size=3, padding=1)

        # === SHARED ENCODER (applied to every slice) ===
        self.downs = nn.ModuleList()
        in_ch = base_channels
        for i in range(1, len(channel_mults)):
            out_ch = base_channels * channel_mults[i]
            self.downs.append(
                nn.ModuleList([TimeConditionedBlock(in_ch, out_ch, time_emb_dim), nn.MaxPool2d(2)])
            )
            in_ch = out_ch

        # === BOTTLENECK: fuse across slices, then collapse to the center ===
        bottleneck_ch = base_channels * channel_mults[-1]
        self.bottleneck1 = TimeConditionedBlock(bottleneck_ch, bottleneck_ch, time_emb_dim)
        self.cross_slice_attn = CrossSliceAttention(bottleneck_ch, heads=attn_heads)
        self.bottleneck2 = TimeConditionedBlock(bottleneck_ch, bottleneck_ch, time_emb_dim)

        # === DECODER (center slice only) ===
        self.ups = nn.ModuleList()
        in_ch = bottleneck_ch
        for i in reversed(range(1, len(channel_mults))):
            skip_ch = base_channels * channel_mults[i]
            out_ch = base_channels * channel_mults[i - 1]
            self.ups.append(
                nn.ModuleList(
                    [
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                        TimeConditionedBlock(in_ch + skip_ch, out_ch, time_emb_dim),
                    ]
                )
            )
            in_ch = out_ch

        self.final_conv = nn.Conv2d(base_channels, 2, kernel_size=1)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Predict the center slice's velocity from a window of neighbouring slices.

        Args:
            x: Cylindrical slice window, shape ``[B, S, 3, H, W]`` with ``S`` odd.
            time: Diffusion time, shape ``[B]``. One value per sample: every slice
                in a window belongs to the same training example and so shares a
                time.

        Returns:
            Velocity for the center slice, shape ``[B, 2, H, W]``.

        Raises:
            ValueError: If ``x`` is not 5D or ``S`` is even (no unique center).
        """
        if x.dim() != 5:
            raise ValueError(
                f"CylindricalUNetCrossSlice expects [B, S, C, H, W], got {tuple(x.shape)}. "
                "Set dataset.num_slices to an odd value > 1 to produce slice windows."
            )

        b, s, c, h, w = x.shape
        if s % 2 == 0:
            raise ValueError(f"num_slices must be odd so a center exists, got S={s}")

        center = s // 2

        t_emb = self.time_mlp(time)
        # The encoder sees B*S folded images, so the time embedding is repeated to
        # match. repeat_interleave (not repeat) keeps each sample's slices adjacent,
        # which is what reshape(b, s, ...) assumes when unfolding later.
        t_emb_slices = t_emb.repeat_interleave(s, dim=0)

        # === Shared encoder over every slice ===
        hidden = self.init_conv(x.reshape(b * s, c, h, w))

        skips = []
        for down_module in self.downs:
            block, pool = down_module[0], down_module[1]
            hidden = block(hidden, t_emb_slices)
            skips.append(hidden)
            hidden = pool(hidden)

        # === Bottleneck: the only place slices talk to each other ===
        hidden = self.bottleneck1(hidden, t_emb_slices)

        _, bott_c, bott_h, bott_w = hidden.shape
        hidden = self.cross_slice_attn(hidden.reshape(b, s, bott_c, bott_h, bott_w))

        # Collapse to the center slice. Everything below is B-sized, so the decoder
        # costs the same as the 2D baseline and t_emb is used unrepeated again.
        hidden = hidden[:, center]
        hidden = self.bottleneck2(hidden, t_emb)

        # === Decoder on the center slice, using the center slice's skips ===
        for up_module, skip in zip(self.ups, reversed(skips), strict=False):
            upsample, block = up_module[0], up_module[1]
            hidden = upsample(hidden)
            skip_center = skip.reshape(b, s, *skip.shape[1:])[:, center]
            hidden = torch.cat([hidden, skip_center], dim=1)
            hidden = block(hidden, t_emb)

        return self.final_conv(hidden)

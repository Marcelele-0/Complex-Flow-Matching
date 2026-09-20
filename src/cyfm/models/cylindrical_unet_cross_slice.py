"""2.5D cylindrical U-Net: shared 2D encoders per slice, fused by cross-slice attention."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from cyfm.core.registry import MODELS
from cyfm.models.cylindrical_unet_attention import (
    SinusoidalPositionEmbeddings,
    TimeConditionedBlock,
)


class CrossSliceAttention(nn.Module):
    """Multi-head attention across the slice axis independently per spatial position.

    Args:
        channels: Feature channel count.
        heads: Number of attention heads.
    """

    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse feature information across slice window.

        Args:
            x: Bottleneck features [B, S, C, H, W].

        Returns:
            Fused features [B, S, C, H, W].
        """
        b, s, c, h, w = x.shape

        normed = self.norm(x.reshape(b * s, c, h, w)).reshape(b, s, c, h * w)
        seq = normed.permute(0, 3, 1, 2).reshape(b * h * w, s, c)
        attn_out, _ = self.mha(seq, seq, seq, need_weights=False)
        attn_out = attn_out.reshape(b, h * w, s, c).permute(0, 2, 3, 1).reshape(b, s, c, h, w)
        return x + attn_out


@MODELS.register("c_unet_cross_slice")
@MODELS.register("cylindrical_unet_cross_slice")
class CylindricalUNetCrossSlice(nn.Module):
    """2.5D U-Net predicting center slice velocity from multi-slice window.

    Args:
        base_channels: Width of the first encoder stage.
        channel_mults: Per-stage width multipliers.
        attn_heads: Heads for cross-slice attention.
        in_channels: Input state channels [B, S, in_channels, H, W].
    """

    def __init__(
        self,
        base_channels: int = 96,
        channel_mults: list[int] | None = None,
        attn_heads: int = 4,
        in_channels: int = 3,
    ) -> None:
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
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Predict center slice velocity field from a window of slices.

        Args:
            x: Slice window tensor [B, S, in_channels, H, W] with S odd.
            time: Diffusion time values [B].

        Returns:
            Center slice velocity field [B, 2, H, W].

        Raises:
            ValueError: If x is not 5D or S is even.
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
        t_emb_slices = t_emb.repeat_interleave(s, dim=0)

        # === Shared encoder over every slice ===
        hidden = self.init_conv(x.reshape(b * s, c, h, w))

        skips = []
        for down_module in self.downs:
            down_list = cast(nn.ModuleList, down_module)
            block = cast(TimeConditionedBlock, down_list[0])
            pool = cast(nn.MaxPool2d, down_list[1])
            hidden = block(hidden, t_emb_slices)
            skips.append(hidden)
            hidden = pool(hidden)

        # === Bottleneck: fuse across slices ===
        hidden = self.bottleneck1(hidden, t_emb_slices)

        _, bott_c, bott_h, bott_w = hidden.shape
        hidden = self.cross_slice_attn(hidden.reshape(b, s, bott_c, bott_h, bott_w))

        # Collapse to center slice
        hidden = hidden[:, center]
        hidden = self.bottleneck2(hidden, t_emb)

        # === Decoder on center slice ===
        for up_module, skip in zip(self.ups, reversed(skips), strict=False):
            up_list = cast(nn.ModuleList, up_module)
            upsample = cast(nn.Upsample, up_list[0])
            block = cast(TimeConditionedBlock, up_list[1])
            hidden = upsample(hidden)
            skip_center = skip.reshape(b, s, *skip.shape[1:])[:, center]
            hidden = torch.cat([hidden, skip_center], dim=1)
            hidden = block(hidden, t_emb)

        return self.final_conv(hidden)

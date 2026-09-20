from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn

from cyfm.core.registry import MODELS


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time position embedding.

    Args:
        dim: Embedding dimension.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Embed scalar diffusion time into feature vectors.

        Args:
            time: Diffusion time values [B].

        Returns:
            Time embeddings [B, dim].
        """
        device = time.device
        half_dim = self.dim // 2
        inv_freq_scale = math.log(10000) / (half_dim - 1)
        inv_freq = torch.exp(torch.arange(half_dim, device=device) * -inv_freq_scale)
        embeddings = time[:, None] * inv_freq[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


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
        self.residual = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.silu(h)

        time_info = self.time_mlp(t_emb)
        time_info = time_info.unsqueeze(-1).unsqueeze(-1)
        h = h + time_info

        h = self.conv2(h)
        h = self.norm2(h)

        return self.silu(h + self.residual(x))


class SelfAttention2d(nn.Module):
    """Multi-head spatial self-attention on 2D feature maps.

    Args:
        channels: Feature channel count.
        heads: Number of attention heads.
    """

    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial self-attention.

        Args:
            x: Input feature maps [B, C, H, W].

        Returns:
            Attention-enhanced feature maps [B, C, H, W].
        """
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).transpose(1, 2)
        norm_x = self.norm(x).view(b, c, h * w).transpose(1, 2)
        attn_out, _ = self.mha(norm_x, norm_x, norm_x)
        out = x_flat + attn_out
        return out.transpose(1, 2).view(b, c, h, w)


@MODELS.register("c_unet_attention")
@MODELS.register("cylindrical_unet_attention")
class CylindricalUNetAttention(nn.Module):
    """Dynamic U-Net with optional bottleneck self-attention for Flow Matching.

    Args:
        base_channels: Width of the first feature level.
        channel_mults: Per-level width multipliers.
        use_attention: Whether to insert self-attention at bottleneck.
        attn_heads: Number of attention heads.
        in_channels: Input state channels [B, in_channels, H, W].
        out_channels: Output velocity channels [B, out_channels, H, W].
    """

    def __init__(
        self,
        base_channels: int = 96,
        channel_mults: list[int] | None = None,
        use_attention: bool = True,
        attn_heads: int = 4,
        in_channels: int = 3,
        out_channels: int = 2,
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

        # === DYNAMIC ENCODER ===
        self.downs = nn.ModuleList()
        in_ch = base_channels
        for i in range(1, len(channel_mults)):
            out_ch = base_channels * channel_mults[i]
            self.downs.append(
                nn.ModuleList([TimeConditionedBlock(in_ch, out_ch, time_emb_dim), nn.MaxPool2d(2)])
            )
            in_ch = out_ch

        # === BOTTLENECK ===
        bottleneck_ch = base_channels * channel_mults[-1]
        self.bottleneck1 = TimeConditionedBlock(bottleneck_ch, bottleneck_ch, time_emb_dim)

        self.use_attention = use_attention
        if use_attention:
            self.attn = SelfAttention2d(bottleneck_ch, heads=attn_heads)

        self.bottleneck2 = TimeConditionedBlock(bottleneck_ch, bottleneck_ch, time_emb_dim)

        # === DYNAMIC DECODER ===
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

        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Predict velocity field for state at given diffusion time.

        Args:
            x: Manifold state tensor [B, in_channels, H, W].
            time: Diffusion time values [B].

        Returns:
            Predicted velocity field [B, out_channels, H, W].
        """
        t_emb = self.time_mlp(time)
        x = self.init_conv(x)

        skips = []
        for down_module in self.downs:
            down_list = cast(nn.ModuleList, down_module)
            block = cast(TimeConditionedBlock, down_list[0])
            pool = cast(nn.MaxPool2d, down_list[1])
            x = block(x, t_emb)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck1(x, t_emb)
        if self.use_attention:
            x = self.attn(x)
        x = self.bottleneck2(x, t_emb)

        for up_module, skip in zip(self.ups, reversed(skips), strict=False):
            up_list = cast(nn.ModuleList, up_module)
            upsample = cast(nn.Upsample, up_list[0])
            block = cast(TimeConditionedBlock, up_list[1])
            x = upsample(x)
            x = torch.cat([x, skip], dim=1)
            x = block(x, t_emb)

        return self.final_conv(x)

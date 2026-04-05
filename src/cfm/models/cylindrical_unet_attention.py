import math

import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Translates a time scalar t into a high-dimensional feature vector (embedding).
    This allows the network to "understand" the progression of time during generation.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimeConditionedBlock(nn.Module):
    """
    Residual Block that takes an image spatial map
    and injects information about the current time step.
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
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
    """
    Standard multi-head self-attention applied to 2D spatial feature maps.
    """

    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).transpose(1, 2)
        norm_x = self.norm(x).view(b, c, h * w).transpose(1, 2)
        attn_out, _ = self.mha(norm_x, norm_x, norm_x)
        out = x_flat + attn_out
        return out.transpose(1, 2).view(b, c, h, w)


class CylindricalUNetAttention(nn.Module):
    """
    Dynamic Cylindrical U-Net built based on configuration parameters.
    Includes optional Self-Attention at the bottleneck.
    """

    def __init__(
        self,
        base_channels: int = 96,
        channel_mults: list | None = None,
        use_attention: bool = True,
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

        # Traverse upwards, reversing the multiplier list
        for i in reversed(range(1, len(channel_mults))):
            skip_ch = base_channels * channel_mults[i]  # Channels from skip-connection
            out_ch = base_channels * channel_mults[i - 1]  # Channels after this block

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
        t_emb = self.time_mlp(time)
        x = self.init_conv(x)

        skips = []
        # Encoder pass
        for down_module in self.downs:
            block, pool = down_module[0], down_module[1]
            x = block(x, t_emb)
            skips.append(x)
            x = pool(x)

        # Bottleneck pass
        x = self.bottleneck1(x, t_emb)
        if self.use_attention:
            x = self.attn(x)
        x = self.bottleneck2(x, t_emb)

        # Decoder pass
        for up_module, skip in zip(self.ups, reversed(skips), strict=False):
            upsample, block = up_module[0], up_module[1]
            x = upsample(x)
            x = torch.cat([x, skip], dim=1)
            x = block(x, t_emb)

        return self.final_conv(x)

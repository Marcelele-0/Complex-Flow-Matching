import math

import torch
import torch.nn as nn

from cfm.core.registry import MODELS


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
        inv_freq_scale = math.log(10000) / (half_dim - 1)
        inv_freq = torch.exp(torch.arange(half_dim, device=device) * -inv_freq_scale)
        embeddings = time[:, None] * inv_freq[None, :]
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

        # Residual connection aligning channels if there was a change
        self.residual = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
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


@MODELS.register("c_unet")
@MODELS.register("cylindrical_unet")
class CylindricalUNet(nn.Module):
    """
    U-Net for flow matching on complex-valued MRI.

    The trunk is geometry-agnostic: only ``in_channels`` records which
    representation is being consumed, so the cylindrical and Euclidean
    experiments run the same architecture and differ by one convolution's input
    width. The name is kept for checkpoint and config compatibility.

    Input: [B, in_channels, H, W]
        cylindrical: 3 channels (Amplitude, p_x, p_y)
        euclidean:   2 channels (Re, Im)
    Output: [B, out_channels, H, W] - the velocity field, 2 channels either way.

    Args:
        base_channels: Width of the first feature level.
        in_channels: Channels of the state, i.e. ``Manifold.state_channels``.
        out_channels: Channels of the velocity, i.e. ``Manifold.velocity_channels``.
    """

    def __init__(self, base_channels: int = 64, in_channels: int = 3, out_channels: int = 2):
        super().__init__()

        time_emb_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # ENCODER (Downsampling, extracting global context)
        self.down1 = TimeConditionedBlock(base_channels, base_channels * 2, time_emb_dim)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = TimeConditionedBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.pool2 = nn.MaxPool2d(2)

        # BOTTLENECK (Deepest representation)
        self.bottleneck = TimeConditionedBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        # DECODER (Upsampling + Skip Connections)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        # Channels: 4*base (from upsample) + 4*base (from skip connection) = 8*base
        self.up_block1 = TimeConditionedBlock(base_channels * 8, base_channels * 2, time_emb_dim)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        # Channels: 2*base (from upsample) + 2*base (from skip connection) = 4*base
        self.up_block2 = TimeConditionedBlock(base_channels * 4, base_channels, time_emb_dim)

        # Final convolution outputting the velocity field
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
        """
        Args:
            x: State tensor from the manifold's transform [B, in_channels, H, W]
            time: Time scalars for each sample in the batch [B]
        """
        t_emb = self.time_mlp(time)

        x = self.init_conv(x)

        # Encoder
        d1 = self.down1(x, t_emb)
        p1 = self.pool1(d1)

        d2 = self.down2(p1, t_emb)
        p2 = self.pool2(d2)

        # Bottleneck
        b = self.bottleneck(p2, t_emb)

        # Decoder with Skip Connections
        u1 = self.up1(b)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.up_block1(u1, t_emb)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.up_block2(u2, t_emb)

        return self.final_conv(u2)

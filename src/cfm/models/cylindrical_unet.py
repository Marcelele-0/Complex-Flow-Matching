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
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.silu = nn.SiLU()

        # Residual connection aligning channels if there was a change
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

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


class CylindricalUNet(nn.Module):
    """
    U-Net tailored for the cylindrical topology (R^+ x S^1).
    Input: [B, 3, H, W] (Amplitude, p_x, p_y)
    Output: [B, 2, H, W] (Amplitude Velocity, Phase Angular Velocity)
    """
    def __init__(self, base_channels: int = 64):
        super().__init__()
        
        time_emb_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Initial convolution (from 3 manifold channels)
        self.init_conv = nn.Conv2d(3, base_channels, kernel_size=3, padding=1)

        # ENCODER (Downsampling, extracting global context)
        self.down1 = TimeConditionedBlock(base_channels, base_channels * 2, time_emb_dim)
        self.pool1 = nn.MaxPool2d(2)
        
        self.down2 = TimeConditionedBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.pool2 = nn.MaxPool2d(2)

        # BOTTLENECK (Deepest representation)
        self.bottleneck = TimeConditionedBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        # DECODER (Upsampling + Skip Connections)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # Channels: 4*base (from upsample) + 4*base (from skip connection) = 8*base
        self.up_block1 = TimeConditionedBlock(base_channels * 8, base_channels * 2, time_emb_dim)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # Channels: 2*base (from upsample) + 2*base (from skip connection) = 4*base
        self.up_block2 = TimeConditionedBlock(base_channels * 4, base_channels, time_emb_dim)

        # Final convolution outputting our 2 velocity vectors
        self.final_conv = nn.Conv2d(base_channels, 2, kernel_size=1)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor from topological projection [B, 3, H, W]
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
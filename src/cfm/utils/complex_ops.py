"""Complex-to-manifold domain mapping transformations."""

from __future__ import annotations

import torch


def complex_to_cylinder(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Map complex tensor to decoupled cylindrical coordinates (magnitude, cos, sin).

    Args:
        z: Complex tensor [B, 1, H, W] or [B, H, W] complex64.
        eps: Epsilon to prevent NaN gradients in zero-magnitude regions.

    Returns:
        Cylindrical tensor [B, 3, H, W] float32 (magnitude, cos(phi), sin(phi)).
    """
    magnitude = torch.abs(z).to(torch.float32)
    phi = torch.angle(z + eps).to(torch.float32)

    p_x = torch.cos(phi)
    p_y = torch.sin(phi)

    return torch.cat([magnitude, p_x, p_y], dim=1)


def cylinder_to_complex(cylinder_tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Map decoupled cylindrical coordinates back to complex tensor.

    Args:
        cylinder_tensor: Cylindrical state [B, 3, H, W] float32.
        eps: Epsilon to avoid division by zero during unit-circle projection.

    Returns:
        Complex tensor [B, 1, H, W] complex64.
    """
    magnitude = cylinder_tensor[:, 0:1, :, :]
    p_x = cylinder_tensor[:, 1:2, :, :]
    p_y = cylinder_tensor[:, 2:3, :, :]

    norm = torch.sqrt(p_x**2 + p_y**2 + eps)
    p_x_projected = p_x / norm
    p_y_projected = p_y / norm

    phi = torch.atan2(p_y_projected, p_x_projected)

    real_part = magnitude * torch.cos(phi)
    imag_part = magnitude * torch.sin(phi)

    return torch.complex(real_part, imag_part)


def complex_to_euclidean(z: torch.Tensor) -> torch.Tensor:
    """Map complex tensor to flat 2-channel Euclidean coordinates (real, imag).

    Args:
        z: Complex tensor [B, 1, H, W] or [B, H, W] complex64.

    Returns:
        Euclidean tensor [B, 2, H, W] float32 (real, imag).
    """
    return torch.cat([z.real.to(torch.float32), z.imag.to(torch.float32)], dim=1)


def euclidean_to_complex(euclidean_tensor: torch.Tensor) -> torch.Tensor:
    """Map flat 2-channel Euclidean coordinates back to complex tensor.

    Args:
        euclidean_tensor: Euclidean state [B, 2, H, W] float32.

    Returns:
        Complex tensor [B, 1, H, W] complex64.
    """
    return torch.complex(euclidean_tensor[:, 0:1, :, :], euclidean_tensor[:, 1:2, :, :])

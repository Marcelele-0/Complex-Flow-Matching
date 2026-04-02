import torch 

def complex_to_cylinder(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert a complex number to a decoupled cylindrical manifold (magnitude and unit-circle phase).
    
    Args:
        z (torch.Tensor): A tensor of shape [batch, 1, H, W] representing complex numbers.
        eps (float): A small value to prevent NaN when calculating the angle of zero-magnitude pixels.
    
    Returns:
        torch.Tensor: A tensor of shape [batch, 3, H, W] of type torch.float32.
                      Channel 0: magnitude
                      Channel 1: phase cos (p_x) with fixed R=1
                      Channel 2: phase sin (p_y) with fixed R=1
    """
    magnitude = torch.abs(z).to(torch.float32)

    # We add eps to prevent NaN gradients in purely empty regions
    phi = torch.angle(z + eps).to(torch.float32)

    # CRITICAL: We DO NOT multiply by magnitude. 
    # This enforces the R=1 topology, preventing phase gradient collapse.
    p_x = torch.cos(phi)
    p_y = torch.sin(phi)

    return torch.cat([magnitude, p_x, p_y], dim=1)


def cylinder_to_complex(cylinder_tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert decoupled cylindrical coordinates back to complex numbers.

    Args:
        cylinder_tensor (torch.Tensor): A tensor of shape [batch, 3, H, W].
        eps (float): A small value to prevent division by zero during projection.

    Returns:
        torch.Tensor: A tensor of shape [batch, 1, H, W] representing the complex numbers (torch.complex64).
    """
    # Fixed slicing syntax
    magnitude = cylinder_tensor[:, 0:1, :, :]
    p_x = cylinder_tensor[:, 1:2, :, :]
    p_y = cylinder_tensor[:, 2:3, :, :]

    # We force R=1 to correct any numerical instability from the ODE solver
    norm = torch.sqrt(p_x ** 2 + p_y ** 2 + eps)
    p_x_projected = p_x / norm
    p_y_projected = p_y / norm

    phi = torch.atan2(p_y_projected, p_x_projected)

    real_part = magnitude * torch.cos(phi)
    imag_part = magnitude * torch.sin(phi)

    return torch.complex(real_part, imag_part)
import math

import pytest
import torch

from cfm.manifolds import CylindricalManifold, EuclideanManifold

MANIFOLDS = [CylindricalManifold, EuclideanManifold]


@pytest.mark.parametrize("manifold_cls", MANIFOLDS)
def test_from_complex_roundtrip_random(manifold_cls: type) -> None:
    manifold = manifold_cls()

    amp = torch.rand(4, 1, 32, 32)
    phi = (torch.rand(4, 1, 32, 32) * 2 - 1) * math.pi
    z = torch.polar(amp, phi)

    state = manifold.from_complex(z)

    assert state.shape == (z.shape[0], manifold.state_channels, z.shape[2], z.shape[3])
    assert state.dtype == torch.float32

    z_out = manifold.to_complex(state)

    assert z_out.shape == z.shape
    assert z_out.dtype == torch.complex64

    torch.testing.assert_close(z_out, z, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("manifold_cls", MANIFOLDS)
def test_from_complex_roundtrip_edge_cases(manifold_cls: type) -> None:
    manifold = manifold_cls()

    # 1. Zero amplitude
    amp_zero = torch.zeros(4, 1, 8, 8)
    phi_rand = (torch.rand(4, 1, 8, 8) * 2 - 1) * math.pi
    z_zero = torch.polar(amp_zero, phi_rand)

    state_zero = manifold.from_complex(z_zero)
    z_zero_out = manifold.to_complex(state_zero)

    torch.testing.assert_close(z_zero_out, z_zero, atol=1e-5, rtol=1e-5)

    # 2. Pi / -Pi phase
    amp_ones = torch.ones(4, 1, 8, 8)
    phi_pi = torch.ones(4, 1, 8, 8) * math.pi
    z_pi = torch.polar(amp_ones, phi_pi)

    state_pi = manifold.from_complex(z_pi)
    z_pi_out = manifold.to_complex(state_pi)

    torch.testing.assert_close(z_pi_out, z_pi, atol=1e-5, rtol=1e-5)

    # 3. Purely imaginary (positive and negative)
    phi_half_pi = (
        torch.tensor([math.pi / 2, -math.pi / 2, math.pi / 2, -math.pi / 2])
        .view(4, 1, 1, 1)
        .expand(-1, -1, 8, 8)
    )
    z_imag = torch.polar(amp_ones, phi_half_pi)

    state_imag = manifold.from_complex(z_imag)
    z_imag_out = manifold.to_complex(state_imag)

    torch.testing.assert_close(z_imag_out, z_imag, atol=1e-5, rtol=1e-5)

import math

import pytest
import torch

from cfm.evaluate import dc_project, integrate_from_t
from cfm.manifolds import CylindricalManifold, EuclideanManifold
from cfm.utils.fft import fft2c

MANIFOLDS = [CylindricalManifold, EuclideanManifold]


@pytest.mark.parametrize("manifold_cls", MANIFOLDS)
def test_dc_project(manifold_cls: type) -> None:
    manifold = manifold_cls()

    # Create random complex state
    amp = torch.rand(2, 1, 32, 32)
    phi = (torch.rand(2, 1, 32, 32) * 2 - 1) * math.pi
    z_state = torch.polar(amp, phi)
    x_state = manifold.from_complex(z_state)

    # Create random measured k-space and a sampling mask
    y_measured = torch.randn(2, 1, 32, 32, dtype=torch.complex64)
    mask = torch.zeros(2, 1, 32, 32)
    # Mask 1D Cartesian pattern
    mask[:, :, :, 10:20] = 1.0

    # Apply DC projection
    x_dc = dc_project(x_state, manifold, y_measured, mask)
    z_dc = manifold.to_complex(x_dc)

    # Verify k-space
    Z_dc = fft2c(z_dc)

    # Where mask == 1, it should match y_measured
    torch.testing.assert_close(Z_dc * mask, y_measured * mask, atol=1e-5, rtol=1e-5)

    # Where mask == 0, it should match the original state's k-space
    Z_state = fft2c(z_state)
    torch.testing.assert_close(Z_dc * (1 - mask), Z_state * (1 - mask), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("manifold_cls", MANIFOLDS)
def test_dc_project_multicoil(manifold_cls: type) -> None:
    """With sensitivity maps, DC enforces the measured multi-coil k-space itself."""
    manifold = manifold_cls()
    b, coils, h, w = 2, 4, 16, 16

    sens_maps = torch.complex(torch.randn(b, coils, h, w), torch.randn(b, coils, h, w))
    sens_maps = sens_maps / torch.sqrt((sens_maps.abs() ** 2).sum(dim=1, keepdim=True) + 1e-8)

    z_state = torch.complex(torch.randn(b, 1, h, w), torch.randn(b, 1, h, w))
    x_state = manifold.from_complex(z_state)

    y_measured = torch.randn(b, coils, h, w, dtype=torch.complex64)
    mask = torch.zeros(b, 1, h, w)
    mask[:, :, :, 4:8] = 1.0

    x_dc = dc_project(x_state, manifold, y_measured, mask, sensitivity_maps=sens_maps)
    z_dc = manifold.to_complex(x_dc)
    assert z_dc.shape == (b, 1, h, w)

    # The projected image, pushed back through the coils, reproduces the measurement
    # on the sampled lines. Only approximately: the single-channel image cannot
    # represent coil content outside the span of the maps.
    residual = (fft2c(sens_maps * z_dc) - y_measured) * mask
    baseline = (fft2c(sens_maps * manifold.to_complex(x_state)) - y_measured) * mask
    assert torch.linalg.norm(residual) < torch.linalg.norm(baseline)


@pytest.mark.parametrize("manifold_cls", MANIFOLDS)
def test_integrate_from_t_with_dc(manifold_cls: type) -> None:
    manifold = manifold_cls()
    solver = manifold.make_solver(num_steps=3)

    x_start = manifold.sample_noise(2, 16, 16, torch.device("cpu"))
    y_measured = torch.randn(2, 1, 16, 16, dtype=torch.complex64)
    mask = torch.zeros(2, 1, 16, 16)
    mask[:, :, :, 4:8] = 1.0

    def dummy_model(x, t):
        return torch.zeros(x.shape[0], manifold.velocity_channels, x.shape[2], x.shape[3])

    # Integrate without DC
    x_no_dc = integrate_from_t(dummy_model, solver, x_start, t_start=0.5)

    # Integrate with DC
    x_with_dc = integrate_from_t(
        dummy_model,
        solver,
        x_start,
        t_start=0.5,
        manifold=manifold,
        y_measured_kspace=y_measured,
        sampling_mask=mask,
        use_dc_projection=True,
    )

    # They should not be equal since DC modified the state
    assert not torch.allclose(x_no_dc, x_with_dc)

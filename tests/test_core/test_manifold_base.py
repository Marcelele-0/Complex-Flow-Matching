"""Tests for BaseManifold interface and manifold implementations (Cylindrical, Euclidean)."""

import pytest
import torch

from cyfm.core.manifold import BaseManifold
from cyfm.manifolds.cylindrical import CylindricalManifold
from cyfm.manifolds.euclidean import EuclideanManifold


def test_cannot_instantiate_abstract_base_manifold() -> None:
    """BaseManifold cannot be instantiated directly due to abstract methods."""
    with pytest.raises(TypeError):
        BaseManifold()  # type: ignore[abstract]


@pytest.mark.parametrize("manifold_cls", [CylindricalManifold, EuclideanManifold])
def test_manifold_interfaces_and_channels(manifold_cls: type[BaseManifold]) -> None:
    """Verify state and velocity channel attributes and solver generation."""
    manifold = manifold_cls()
    assert isinstance(manifold, BaseManifold)
    assert manifold.velocity_channels == 2
    assert manifold.state_channels in (2, 3)

    solver = manifold.make_solver(num_steps=10)
    assert solver.num_steps == 10


@pytest.mark.parametrize("manifold_cls", [CylindricalManifold, EuclideanManifold])
def test_exp_and_log_map_inversion(manifold_cls: type[BaseManifold]) -> None:
    """exp_map(x_0, log_map(x_0, x_1)) should recover x_1."""
    manifold = manifold_cls()
    b, h, w = 2, 8, 8
    device = torch.device("cpu")

    # Sample valid points on manifold
    x_0 = manifold.sample_noise(b, h, w, device)
    x_1 = manifold.sample_noise(b, h, w, device)

    # Compute tangent vector from x_0 to x_1
    v = manifold.log_map(x_0, x_1)
    assert v.shape == (b, 2, h, w)

    # Shoot along geodesic via exp_map
    x_rec = manifold.exp_map(x_0, v)
    assert x_rec.shape == x_0.shape

    if isinstance(manifold, EuclideanManifold):
        torch.testing.assert_close(x_rec, x_1, atol=1e-5, rtol=1e-5)
    elif isinstance(manifold, CylindricalManifold):
        # Amplitude matches
        torch.testing.assert_close(x_rec[:, 0:1], x_1[:, 0:1], atol=1e-5, rtol=1e-5)
        # Cos/Sin phase matches
        torch.testing.assert_close(x_rec[:, 1:3], x_1[:, 1:3], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("manifold_cls", [CylindricalManifold, EuclideanManifold])
def test_geodesic_path_boundary_conditions(manifold_cls: type[BaseManifold]) -> None:
    """geodesic_path at t=0 returns x_0, at t=1 returns x_1."""
    manifold = manifold_cls()
    b, h, w = 2, 8, 8
    device = torch.device("cpu")

    x_0 = manifold.sample_noise(b, h, w, device)
    x_1 = manifold.sample_noise(b, h, w, device)

    t_0 = torch.zeros(b, 1, 1, 1, device=device)
    t_1 = torch.ones(b, 1, 1, 1, device=device)

    path_0 = manifold.geodesic_path(x_0, x_1, t_0)
    path_1 = manifold.geodesic_path(x_0, x_1, t_1)

    torch.testing.assert_close(path_0, x_0, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(path_1, x_1, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("manifold_cls", [CylindricalManifold, EuclideanManifold])
def test_bridge_consistency_with_geodesic_and_velocity(manifold_cls: type[BaseManifold]) -> None:
    """bridge(x_0, x_1, t) output matches geodesic_path and target_velocity."""
    manifold = manifold_cls()
    b, h, w = 2, 8, 8
    device = torch.device("cpu")

    x_0 = manifold.sample_noise(b, h, w, device)
    x_1 = manifold.sample_noise(b, h, w, device)
    t = torch.full((b, 1, 1, 1), 0.35, device=device)

    x_t_bridge, u_bridge = manifold.bridge(x_0, x_1, t)
    x_t_geo = manifold.geodesic_path(x_0, x_1, t)
    u_target = manifold.target_velocity(x_0, x_1, t)

    torch.testing.assert_close(x_t_bridge, x_t_geo, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(u_bridge, u_target, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("manifold_cls", [CylindricalManifold, EuclideanManifold])
def test_metric_tensor(manifold_cls: type[BaseManifold]) -> None:
    """Metric tensor returns expected shape and positive values."""
    manifold = manifold_cls()
    x = torch.randn(2, manifold.state_channels, 16, 16)
    g = manifold.metric_tensor(x)
    assert g.shape == (2, manifold.velocity_channels, 16, 16)
    assert (g > 0).all()


@pytest.mark.parametrize("manifold_cls", [CylindricalManifold, EuclideanManifold])
def test_complex_domain_round_trip(manifold_cls: type[BaseManifold]) -> None:
    """Converting from complex to manifold state and back preserves complex values."""
    manifold = manifold_cls()
    # Complex tensor normalized with amplitude <= 1
    z = torch.complex(torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16))
    z = z / (z.abs().max() + 1e-6)

    state = manifold.from_complex(z)
    assert state.shape == (2, manifold.state_channels, 16, 16)

    z_rec = manifold.to_complex(state)
    assert z_rec.shape == (2, 1, 16, 16)
    assert z_rec.is_complex()
    torch.testing.assert_close(z_rec, z, atol=1e-5, rtol=1e-5)

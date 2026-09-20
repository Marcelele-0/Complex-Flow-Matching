import math

import torch

from cyfm.flow.bridge import GeodesicFlowBridge


def test_bridge_shapes() -> None:
    """Checks that the bridge returns tensors with the expected shapes."""
    bridge = GeodesicFlowBridge()
    batch, height, width = 2, 8, 8
    noise = torch.randn(batch, 3, height, width)
    data = torch.randn(batch, 3, height, width)
    t = torch.rand(batch, 1, 1, 1)

    cyl_t, target_v = bridge.forward(noise, data, t)

    assert cyl_t.shape == (batch, 3, height, width), "State tensor shape mismatch"
    assert target_v.shape == (batch, 2, height, width), "Velocity tensor shape mismatch"


def test_bridge_boundaries() -> None:
    """Checks that interpolation hits the endpoints at t=0 and t=1."""
    bridge = GeodesicFlowBridge()
    noise = torch.randn(1, 3, 4, 4)
    data = torch.randn(1, 3, 4, 4)

    t_0 = torch.zeros(1, 1, 1, 1)
    cyl_0, _ = bridge.forward(noise, data, t_0)

    assert torch.allclose(
        cyl_0[:, 0:1],
        noise[:, 0:1],
    ), "At t=0, amplitude should match noise exactly"

    t_1 = torch.ones(1, 1, 1, 1)
    cyl_1, _ = bridge.forward(noise, data, t_1)

    assert torch.allclose(
        cyl_1[:, 0:1],
        data[:, 0:1],
    ), "At t=1, amplitude should match data exactly"


def test_shortest_angular_diff() -> None:
    """CRITICAL: Checks that phase velocity takes the shortest path on the circle."""
    bridge = GeodesicFlowBridge()

    start = torch.tensor([350.0 * math.pi / 180.0])
    end = torch.tensor([10.0 * math.pi / 180.0])

    diff = bridge.get_shortest_angular_diff(start, end)
    expected_diff = torch.tensor([20.0 * math.pi / 180.0])

    assert torch.allclose(
        diff,
        expected_diff,
        atol=1e-6,
    ), "Angular difference failed shortest path logic"

import torch

from cfm.flow.bridge import GeodesicFlowBridge
from cfm.flow.euclidean_bridge import LinearFlowBridge


def _euclidean_state(batch: int, h: int, w: int, seed: int = 0) -> torch.Tensor:
    """A valid Euclidean state: modulus in [0, 1], phase anywhere."""
    torch.manual_seed(seed)
    amp = torch.rand(batch, 1, h, w)
    phi = torch.rand(batch, 1, h, w) * 2 * torch.pi
    return torch.cat([amp * torch.cos(phi), amp * torch.sin(phi)], dim=1)


def test_bridge_shapes() -> None:
    """Checks that the bridge returns tensors with the expected shapes."""
    bridge = LinearFlowBridge()
    batch, height, width = 2, 8, 8
    noise = _euclidean_state(batch, height, width, seed=0)
    data = _euclidean_state(batch, height, width, seed=1)
    t = torch.rand(batch, 1, 1, 1)

    euc_t, target_v = bridge.forward(noise, data, t)

    assert euc_t.shape == (batch, 2, height, width), "State tensor shape mismatch"
    assert target_v.shape == (batch, 2, height, width), "Velocity tensor shape mismatch"


def test_bridge_boundaries() -> None:
    """Checks that interpolation hits the endpoints at t=0 and t=1."""
    bridge = LinearFlowBridge()
    noise = _euclidean_state(1, 4, 4, seed=2)
    data = _euclidean_state(1, 4, 4, seed=3)

    euc_0, _ = bridge.forward(noise, data, torch.zeros(1, 1, 1, 1))
    assert torch.allclose(euc_0, noise), "At t=0 the state should be the noise exactly"

    euc_1, _ = bridge.forward(noise, data, torch.ones(1, 1, 1, 1))
    assert torch.allclose(euc_1, data, atol=1e-6), "At t=1 the state should be the data"


def test_velocity_is_the_plain_difference() -> None:
    """CRITICAL: the target must be x_1 - x_0 with no wrapping of any kind.

    This is the property the whole baseline rests on. If anything here ever
    started taking a shortest path, the "Euclidean" arm would stop being the
    thing the paper claims to compare against.
    """
    bridge = LinearFlowBridge()
    noise = _euclidean_state(2, 4, 4, seed=4)
    data = _euclidean_state(2, 4, 4, seed=5)

    _, target_v = bridge.forward(noise, data, torch.rand(2, 1, 1, 1))

    assert torch.equal(target_v, data - noise)


def test_velocity_does_not_depend_on_t() -> None:
    """The straight path has a constant conditional velocity, by construction."""
    bridge = LinearFlowBridge()
    noise = _euclidean_state(1, 4, 4, seed=6)
    data = _euclidean_state(1, 4, 4, seed=7)

    _, v_early = bridge.forward(noise, data, torch.full((1, 1, 1, 1), 0.1))
    _, v_late = bridge.forward(noise, data, torch.full((1, 1, 1, 1), 0.9))

    assert torch.equal(v_early, v_late)


def test_state_is_a_straight_line_not_a_geodesic() -> None:
    """The two bridges must genuinely disagree on the same complex endpoints.

    Take a point at angle -3pi/4 and one at +3pi/4: the geodesic crosses the
    branch cut in one direction, the straight line passes near the origin
    instead. If this test ever passed by both sides agreeing, one of the two
    bridges would have quietly become the other.
    """
    angle_0, angle_1 = -3 * torch.pi / 4, 3 * torch.pi / 4
    amp = 1.0
    t = torch.full((1, 1, 1, 1), 0.5)

    cyl_0 = torch.tensor(
        [
            [
                [[amp]],
                [[float(torch.cos(torch.tensor(angle_0)))]],
                [[float(torch.sin(torch.tensor(angle_0)))]],
            ]
        ]
    )
    cyl_1 = torch.tensor(
        [
            [
                [[amp]],
                [[float(torch.cos(torch.tensor(angle_1)))]],
                [[float(torch.sin(torch.tensor(angle_1)))]],
            ]
        ]
    )
    cyl_t, _ = GeodesicFlowBridge().forward(cyl_noise=cyl_0, cyl_data=cyl_1, t=t)

    # The geodesic wraps through pi, so the midpoint sits at |angle| = pi.
    geodesic_angle = torch.atan2(cyl_t[:, 2:3], cyl_t[:, 1:2])
    assert torch.allclose(geodesic_angle.abs(), torch.tensor(torch.pi), atol=1e-5)

    euc_0 = torch.tensor(
        [[[[float(torch.cos(torch.tensor(angle_0)))]], [[float(torch.sin(torch.tensor(angle_0)))]]]]
    )
    euc_1 = torch.tensor(
        [[[[float(torch.cos(torch.tensor(angle_1)))]], [[float(torch.sin(torch.tensor(angle_1)))]]]]
    )
    euc_t, _ = LinearFlowBridge().forward(euc_0, euc_1, t)

    # The chord's midpoint has zero imaginary part and a modulus well below 1:
    # it cuts across the disc rather than following its rim.
    assert torch.allclose(euc_t[:, 1:2], torch.zeros(1), atol=1e-6)
    modulus = torch.sqrt(euc_t[:, 0:1] ** 2 + euc_t[:, 1:2] ** 2)
    assert torch.all(modulus < 0.9)

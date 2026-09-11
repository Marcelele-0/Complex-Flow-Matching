import math

import pytest
import torch

from cfm.flow.torus_math import DecoupledCylindricalLoss


def test_loss_shapes_and_types() -> None:
    """Checks that loss returns four scalar float32 tensors."""
    criterion = DecoupledCylindricalLoss()
    pred = torch.randn(4, 2, 16, 16)
    target = torch.randn(4, 2, 16, 16)

    total, l_amp, l_phase, l_hf = criterion(pred, target)

    assert total.dim() == 0
    assert l_amp.dim() == 0
    assert l_phase.dim() == 0
    assert l_hf.dim() == 0
    assert total.dtype == torch.float32


def test_rejects_three_channel_velocity() -> None:
    """The old 3-channel velocity layout must fail loudly, not silently truncate."""
    criterion = DecoupledCylindricalLoss()
    pred_3ch = torch.zeros(1, 3, 4, 4)
    target_3ch = torch.zeros(1, 3, 4, 4)

    with pytest.raises(ValueError):
        criterion(pred_3ch, target_3ch)


def test_amplitude_loss_variants() -> None:
    """Checks L1 and squared-error calculations for the amplitude channel."""
    pred = torch.zeros(1, 2, 1, 1)
    target = torch.zeros(1, 2, 1, 1)

    # Setup an absolute error of 3.0
    pred[0, 0, 0, 0] = 5.0
    target[0, 0, 0, 0] = 2.0

    # Test L1 (|3.0| = 3.0)
    crit_l1 = DecoupledCylindricalLoss(amp_loss_type="l1", lambda_phase=0.0)
    _, l_amp_l1, _, _ = crit_l1(pred, target)
    assert torch.isclose(l_amp_l1, torch.tensor(3.0)), "L1 amplitude loss failed"

    # Test L2 (3.0^2 = 9.0)
    crit_l2 = DecoupledCylindricalLoss(amp_loss_type="l2", lambda_phase=0.0)
    _, l_amp_l2, _, _ = crit_l2(pred, target)
    assert torch.isclose(l_amp_l2, torch.tensor(9.0)), "L2 amplitude loss failed"

    # "mse" is an alias for "l2": same squared error of 9.0
    crit_mse = DecoupledCylindricalLoss(amp_loss_type="mse", lambda_phase=0.0)
    _, l_amp_mse, _, _ = crit_mse(pred, target)
    assert torch.isclose(l_amp_mse, torch.tensor(9.0)), "MSE amplitude loss failed"


def test_phase_loss_l1_tangent_space() -> None:
    """CRITICAL: v_phase is an angular velocity (tangent space of S^1), NOT an angle.

    Plain L1 applies with no circular wrapping: a full-turn velocity of 2*pi
    rad/unit-time is genuinely different from standing still - the ODE solver
    integrates it literally. The shortest-path wrap belongs to the bridge
    (see test_bridge.py::test_shortest_angular_diff), not the loss.
    """
    crit_l1 = DecoupledCylindricalLoss(phase_loss_type="l1", lambda_phase=1.0)
    pred = torch.zeros(1, 2, 1, 1)
    target = torch.zeros(1, 2, 1, 1)

    # Case 1: A full-turn velocity vs zero velocity is a 2*pi error, NOT zero
    pred[0, 1, 0, 0] = 2 * math.pi
    target[0, 1, 0, 0] = 0.0
    _, _, l_phase1, _ = crit_l1(pred, target)
    assert torch.isclose(l_phase1, torch.tensor(2 * math.pi), atol=1e-6)

    # Case 2: Plain magnitude difference of angular velocities
    pred[0, 1, 0, 0] = 1.5
    target[0, 1, 0, 0] = 0.5
    _, _, l_phase2, _ = crit_l1(pred, target)
    assert torch.isclose(l_phase2, torch.tensor(1.0), atol=1e-6)


def test_phase_loss_cosine() -> None:
    """Checks the mathematical boundaries of the Cosine Angular Loss.

    "cosine" is the explicit opt-in wrapping variant (1 - cos(diff)) kept
    for ablations against the default tangent-space L1.
    """
    crit_cos = DecoupledCylindricalLoss(phase_loss_type="cosine", lambda_phase=1.0)
    pred = torch.zeros(1, 2, 1, 1)
    target = torch.zeros(1, 2, 1, 1)

    # 0 difference -> 1 - cos(0) = 0.0
    pred[0, 1, 0, 0] = 0.0
    target[0, 1, 0, 0] = 0.0
    _, _, l_cos_0, _ = crit_cos(pred, target)
    assert torch.isclose(l_cos_0, torch.tensor(0.0), atol=1e-6)

    # 2*pi difference -> 1 - cos(2pi) = 0.0 (Wrapping test)
    pred[0, 1, 0, 0] = 2 * math.pi
    target[0, 1, 0, 0] = 0.0
    _, _, l_cos_2pi, _ = crit_cos(pred, target)
    assert torch.isclose(l_cos_2pi, torch.tensor(0.0), atol=1e-6)

    # pi difference -> 1 - cos(pi) = 1 - (-1) = 2.0 (Maximum possible error)
    pred[0, 1, 0, 0] = math.pi
    target[0, 1, 0, 0] = 0.0
    _, _, l_cos_pi, _ = crit_cos(pred, target)
    assert torch.isclose(l_cos_pi, torch.tensor(2.0), atol=1e-6)


def test_phase_amplitude_weighting_toggle() -> None:
    """A phase error at a zero-amplitude pixel counts only when weighting is off.

    Two pixels with clean amplitudes 0 and 1, phase error 1.0 at the dark one:
    weighted -> 0.0 (weight 0), unweighted -> 0.5 (mean over two pixels), which
    must equal the loss computed without ``target_x1`` at all.
    """
    pred = torch.zeros(1, 2, 1, 2)
    target = torch.zeros(1, 2, 1, 2)
    pred[0, 1, 0, 0] = 1.0  # phase error at the dark pixel only

    target_x1 = torch.zeros(1, 3, 1, 2)
    target_x1[0, 0, 0, 1] = 1.0  # amplitude: dark pixel 0, bright pixel 1

    weighted = DecoupledCylindricalLoss(phase_amplitude_weighting=True)
    unweighted = DecoupledCylindricalLoss(phase_amplitude_weighting=False)

    _, _, l_phi_on, _ = weighted(pred, target, target_x1)
    _, _, l_phi_off, _ = unweighted(pred, target, target_x1)
    _, _, l_phi_no_x1, _ = unweighted(pred, target)

    assert torch.isclose(l_phi_on, torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(l_phi_off, torch.tensor(0.5), atol=1e-6)
    assert torch.isclose(l_phi_off, l_phi_no_x1)


def test_hf_loss_disabled_and_enabled() -> None:
    """loss_hf must be zero when lambda_hf=0 and positive on a nonzero error otherwise."""
    pred = torch.ones(1, 2, 8, 8)
    target = torch.zeros(1, 2, 8, 8)

    crit_off = DecoupledCylindricalLoss(lambda_hf=0.0)
    _, _, _, l_hf_off = crit_off(pred, target)
    assert torch.isclose(l_hf_off, torch.tensor(0.0))

    crit_on = DecoupledCylindricalLoss(lambda_hf=0.5)
    total_on, l_amp, l_phi, l_hf_on = crit_on(pred, target)
    assert l_hf_on > 0.0
    # The HF term must actually contribute to the total
    assert torch.isclose(total_on, l_amp + l_phi + 0.5 * l_hf_on, atol=1e-6)


def test_invalid_loss_config() -> None:
    """Ensures the class catches bad configuration strings immediately."""
    with pytest.raises(ValueError):
        DecoupledCylindricalLoss(amp_loss_type="magic_loss")

    with pytest.raises(ValueError):
        DecoupledCylindricalLoss(phase_loss_type="euclidean")

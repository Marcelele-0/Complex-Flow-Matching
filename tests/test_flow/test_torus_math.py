import math

import pytest
import torch

from cfm.flow.torus_math import DecoupledCylindricalLoss


def test_loss_shapes_and_types() -> None:
    """Checks that loss returns three scalar float32 tensors."""
    criterion = DecoupledCylindricalLoss()
    pred = torch.randn(4, 2, 16, 16)
    target = torch.randn(4, 2, 16, 16)

    total, l_amp, l_phase = criterion(pred, target)

    assert total.dim() == 0
    assert l_amp.dim() == 0
    assert l_phase.dim() == 0
    assert total.dtype == torch.float32


def test_amplitude_loss_variants() -> None:
    """Checks L1, L2, and Huber calculations for the amplitude channel."""
    pred = torch.zeros(1, 2, 1, 1)
    target = torch.zeros(1, 2, 1, 1)

    # Setup an absolute error of 3.0
    pred[0, 0, 0, 0] = 5.0
    target[0, 0, 0, 0] = 2.0

    # Test L1 (|3.0| = 3.0)
    crit_l1 = DecoupledCylindricalLoss(amp_loss_type="l1", lambda_phase=0.0)
    _, l_amp_l1, _ = crit_l1(pred, target)
    assert torch.isclose(l_amp_l1, torch.tensor(3.0)), "L1 amplitude loss failed"

    # Test L2 (3.0^2 = 9.0)
    crit_l2 = DecoupledCylindricalLoss(amp_loss_type="l2", lambda_phase=0.0)
    _, l_amp_l2, _ = crit_l2(pred, target)
    assert torch.isclose(l_amp_l2, torch.tensor(9.0)), "L2 amplitude loss failed"

    # Test Huber (with delta 1.0, error > delta, so it falls back to L1-like linear scale)
    # Expected: delta * (|err| - 0.5 * delta) = 1.0 * (3.0 - 0.5) = 2.5
    crit_huber = DecoupledCylindricalLoss(amp_loss_type="huber", huber_delta=1.0, lambda_phase=0.0)
    _, l_amp_huber, _ = crit_huber(pred, target)
    assert torch.isclose(l_amp_huber, torch.tensor(2.5)), "Huber amplitude loss failed"


def test_phase_loss_l1_topology_wrap() -> None:
    """CRITICAL: Checks shortest circular phase distance behavior for L1."""
    crit_l1 = DecoupledCylindricalLoss(phase_loss_type="l1", lambda_phase=1.0)
    pred = torch.zeros(1, 2, 1, 1)
    target = torch.zeros(1, 2, 1, 1)

    # Case 1: Full turn should be zero error
    pred[0, 1, 0, 0] = 2 * math.pi
    target[0, 1, 0, 0] = 0.0
    _, _, l_phase1 = crit_l1(pred, target)
    assert torch.isclose(l_phase1, torch.tensor(0.0), atol=1e-6)

    # Case 2: 350 deg vs 10 deg should give shortest path: 20 deg
    pred[0, 1, 0, 0] = 350 * (math.pi / 180)
    target[0, 1, 0, 0] = 10 * (math.pi / 180)
    _, _, l_phase2 = crit_l1(pred, target)
    expected_rad = 20 * (math.pi / 180)
    assert torch.isclose(l_phase2, torch.tensor(expected_rad), atol=1e-6)


def test_phase_loss_cosine() -> None:
    """Checks the mathematical boundaries of the Cosine Angular Loss."""
    crit_cos = DecoupledCylindricalLoss(phase_loss_type="cosine", lambda_phase=1.0)
    pred = torch.zeros(1, 2, 1, 1)
    target = torch.zeros(1, 2, 1, 1)

    # 0 difference -> 1 - cos(0) = 0.0
    pred[0, 1, 0, 0] = 0.0
    target[0, 1, 0, 0] = 0.0
    _, _, l_cos_0 = crit_cos(pred, target)
    assert torch.isclose(l_cos_0, torch.tensor(0.0), atol=1e-6)

    # 2*pi difference -> 1 - cos(2pi) = 0.0 (Wrapping test)
    pred[0, 1, 0, 0] = 2 * math.pi
    target[0, 1, 0, 0] = 0.0
    _, _, l_cos_2pi = crit_cos(pred, target)
    assert torch.isclose(l_cos_2pi, torch.tensor(0.0), atol=1e-6)

    # pi difference -> 1 - cos(pi) = 1 - (-1) = 2.0 (Maximum possible error)
    pred[0, 1, 0, 0] = math.pi
    target[0, 1, 0, 0] = 0.0
    _, _, l_cos_pi = crit_cos(pred, target)
    assert torch.isclose(l_cos_pi, torch.tensor(2.0), atol=1e-6)


def test_invalid_loss_config() -> None:
    """Ensures the class catches bad configuration strings immediately."""
    with pytest.raises(ValueError):
        DecoupledCylindricalLoss(amp_loss_type="magic_loss")

    with pytest.raises(ValueError):
        DecoupledCylindricalLoss(phase_loss_type="euclidean")

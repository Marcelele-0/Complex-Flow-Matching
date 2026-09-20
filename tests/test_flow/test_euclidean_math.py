import pytest
import torch

from cyfm.flow.euclidean_math import EuclideanVelocityLoss
from cyfm.flow.torus_math import DecoupledCylindricalLoss


class TestValidation:
    def test_rejects_unknown_loss_type(self) -> None:
        with pytest.raises(ValueError, match="loss_type must be one of"):
            EuclideanVelocityLoss(loss_type="huber")

    @pytest.mark.parametrize("loss_type", ["l1", "l2", "mse"])
    def test_accepts_the_same_vocabulary_as_the_cylindrical_loss(self, loss_type: str) -> None:
        """Both losses must speak the same config language, or a sweep silently skips one."""
        EuclideanVelocityLoss(loss_type=loss_type)
        DecoupledCylindricalLoss(amp_loss_type=loss_type)

    def test_rejects_three_channel_velocity(self) -> None:
        """A cylindrical state reaching this loss must fail loudly, not broadcast."""
        criterion = EuclideanVelocityLoss()
        with pytest.raises(ValueError, match="Expected 2-channel velocity"):
            criterion(torch.zeros(1, 3, 4, 4), torch.zeros(1, 3, 4, 4))


class TestLossValues:
    def test_l1_is_the_mean_absolute_error_over_both_channels(self) -> None:
        criterion = EuclideanVelocityLoss(loss_type="l1")
        pred = torch.tensor([[[[1.0]], [[-1.0]]]])
        target = torch.zeros_like(pred)

        total, loss_vel, loss_hf = criterion(pred, target)

        assert loss_vel.item() == pytest.approx(1.0)
        assert loss_hf.item() == 0.0
        assert total.item() == pytest.approx(1.0)

    def test_mse_squares_the_error(self) -> None:
        criterion = EuclideanVelocityLoss(loss_type="mse")
        pred = torch.full((1, 2, 4, 4), 3.0)
        target = torch.zeros_like(pred)

        _, loss_vel, _ = criterion(pred, target)

        assert loss_vel.item() == pytest.approx(9.0)

    def test_perfect_prediction_is_zero(self) -> None:
        criterion = EuclideanVelocityLoss()
        v = torch.randn(2, 2, 8, 8)

        total, _, _ = criterion(v, v)

        assert total.item() == pytest.approx(0.0, abs=1e-7)

    def test_target_x1_is_accepted_and_changes_nothing(self) -> None:
        """No amplitude mask in flat space: the argument exists only for interface parity.

        The cylindrical loss masks its phase term because phase is undefined in
        air. Re/Im has no phase channel, so masking here would import a
        cylindrical-specific correction into the baseline. This test pins that
        decision so it cannot drift in silently.
        """
        criterion = EuclideanVelocityLoss()
        pred = torch.randn(2, 2, 8, 8)
        target = torch.randn(2, 2, 8, 8)
        x_1 = torch.rand(2, 2, 8, 8)

        without, _, _ = criterion(pred, target)
        with_x1, _, _ = criterion(pred, target, target_x1=x_1)

        assert torch.equal(without, with_x1)

    def test_channels_are_weighted_equally(self) -> None:
        """Re and Im share a unit and a scale, so neither may be favoured."""
        criterion = EuclideanVelocityLoss()
        target = torch.zeros(1, 2, 4, 4)

        real_error = torch.zeros(1, 2, 4, 4)
        real_error[:, 0] = 1.0
        imag_error = torch.zeros(1, 2, 4, 4)
        imag_error[:, 1] = 1.0

        assert criterion(real_error, target)[0].item() == pytest.approx(
            criterion(imag_error, target)[0].item()
        )


class TestHighFrequencyBoost:
    def test_disabled_by_default(self) -> None:
        criterion = EuclideanVelocityLoss()
        _, _, loss_hf = criterion(torch.randn(1, 2, 8, 8), torch.zeros(1, 2, 8, 8))
        assert loss_hf.item() == 0.0

    def test_enabled_boost_adds_a_positive_term(self) -> None:
        criterion = EuclideanVelocityLoss(lambda_hf=1.0, hf_boost_factor=4.0)
        pred = torch.randn(1, 2, 8, 8)
        target = torch.zeros_like(pred)

        total, loss_vel, loss_hf = criterion(pred, target)

        assert loss_hf.item() > 0.0
        assert total.item() == pytest.approx((loss_vel + loss_hf).item(), rel=1e-6)

    def test_hf_term_is_identical_to_the_cylindrical_one(self) -> None:
        """A HF ablation must mean the same thing on both geometries.

        Same velocity error, same boost factor, same number: the two losses share
        cyfm.flow.spectral rather than each spelling the FFT out.
        """
        error = torch.randn(2, 2, 16, 16)
        target = torch.zeros_like(error)

        euclidean = EuclideanVelocityLoss(lambda_hf=1.0, hf_boost_factor=4.0)
        cylindrical = DecoupledCylindricalLoss(lambda_hf=1.0, hf_boost_factor=4.0)

        _, _, euc_hf = euclidean(error, target)
        _, _, _, cyl_hf = cylindrical(error, target)

        assert torch.equal(euc_hf, cyl_hf)

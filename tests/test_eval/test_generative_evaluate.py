"""Unit tests for the evaluation entry point: reporting, straightness, reference domain.

The distributional metrics themselves are tested in tests/test_utils/test_metrics.py.
"""

import math

import pytest
import torch

from cyfm.config.adapters import manifold_from_config
from cyfm.data.toy import CylinderToyIIDDataset
from cyfm.evaluate import assert_training_domain, format_table
from cyfm.manifolds.euclidean import EuclideanManifold


def _generator(seed: int = 0) -> torch.Generator:
    """A CPU generator pinned to ``seed``."""
    return torch.Generator().manual_seed(seed)


# --- Reporting ---


def test_format_table_renders_one_row_per_step_count() -> None:
    """The sweep table is the artefact the NFE claim is read off."""
    rendered = format_table([(1, {"a": 0.5}), (100, {"a": 0.25})], ["a"])
    lines = rendered.splitlines()
    assert lines[0].split() == ["steps", "a"]
    assert lines[2].split() == ["1", "0.50000"]
    assert lines[3].split() == ["100", "0.25000"]


# --- Straightness is scored under the pairing training used ---


def _straightness_setup() -> tuple[object, object, torch.Tensor]:
    """A small cylindrical model and a data batch in its state representation."""
    from omegaconf import OmegaConf

    from cyfm.models.mlp import PointwiseVelocityMLP

    torch.manual_seed(0)
    manifold = manifold_from_config(
        OmegaConf.create({"manifold": {"name": "cylindrical"}, "training": {"loss": {}}})
    )
    model = PointwiseVelocityMLP(in_channels=3, base_channels=16, depth=1).eval()
    dataset = CylinderToyIIDDataset(size=8, crop_size=(8, 8))
    data = manifold.from_complex(torch.stack([dataset[i] for i in range(8)]))
    return manifold, model, data


def test_independent_coupling_reproduces_the_uncoupled_straightness() -> None:
    """Passing the baseline coupling explicitly must not change the number."""
    from cyfm.evaluate import straightness
    from cyfm.flow.couplings import IndependentCoupling

    manifold, model, data = _straightness_setup()
    cpu = torch.device("cpu")
    bare = straightness(model, manifold, data, cpu, _generator(3))  # type: ignore[arg-type]
    explicit = straightness(
        model,  # type: ignore[arg-type]
        manifold,
        data,
        cpu,
        _generator(3),
        IndependentCoupling(),
    )
    assert bare == explicit


def test_ot_coupling_changes_the_pairs_that_are_scored() -> None:
    """An OT-trained model is scored on OT pairs, which are not the drawn pairs."""
    from cyfm.evaluate import straightness
    from cyfm.flow.couplings import OptimalTransportCoupling

    manifold, model, data = _straightness_setup()
    cpu = torch.device("cpu")
    independent = straightness(model, manifold, data, cpu, _generator(4))  # type: ignore[arg-type]
    coupled = straightness(
        model,  # type: ignore[arg-type]
        manifold,
        data,
        cpu,
        _generator(4),
        OptimalTransportCoupling(),
    )
    assert math.isfinite(coupled)
    assert coupled != independent


# --- The reference lives in the training domain ---


@pytest.mark.parametrize("geometry", ["cylindrical", "euclidean"])
def test_reference_goes_through_the_training_normalisation(geometry: str) -> None:
    """Every reference field ends up with peak modulus one, as training sees it.

    Scoring against raw fields penalised a model for the normalisation itself.
    """
    from omegaconf import OmegaConf

    from cyfm.evaluate import training_pipeline

    manifold = manifold_from_config(
        OmegaConf.create({"manifold": {"name": geometry}, "training": {"loss": {}}})
    )
    pipeline = training_pipeline({"crop_size": [16, 16]}, manifold)
    dataset = CylinderToyIIDDataset(size=4, crop_size=(16, 16))

    for index in range(4):
        raw = dataset[index]
        field = manifold.to_complex(pipeline(raw.clone()).unsqueeze(0))
        assert field.shape == (1, 1, 16, 16)
        assert float(field.abs().max()) == pytest.approx(1.0, abs=1e-5)
        assert float(raw.abs().max()) > 1.0


def test_domain_guard_accepts_a_normalised_batch() -> None:
    """The guard returns the observed peak, which the run's record reports."""
    fields = torch.ones(3, 1, 8, 8, dtype=torch.complex64)
    assert assert_training_domain(fields) == pytest.approx(1.0)


def test_domain_guard_rejects_un_normalised_fields() -> None:
    """The defect it exists for: raw fields scored as if they were normalised."""
    raw = torch.full((2, 1, 8, 8), 3.0 + 0.0j, dtype=torch.complex64)
    with pytest.raises(ValueError, match="not in the training domain"):
        assert_training_domain(raw)


def test_domain_guard_allows_a_peak_removed_by_the_crop() -> None:
    """One-sided on purpose.

    Normalisation runs before the crop, so a field whose brightest pixel was
    cropped away peaks below one and is perfectly valid. A two-sided check would
    fail those and push someone to normalise after cropping, which would change
    the protocol.
    """
    dim = torch.full((2, 1, 8, 8), 0.4 + 0.0j, dtype=torch.complex64)
    assert assert_training_domain(dim) == pytest.approx(0.4)


def test_probe_accumulates_across_batches_of_different_sizes() -> None:
    """num_fields need not divide batch_size, so the last batch is short.

    Reducing a short batch elementwise against the previous one would raise, or
    silently pair two unrelated samples; the extrema are per sample, so batches
    concatenate.
    """
    from cyfm.metrics import AngularVelocityProbe

    probe = AngularVelocityProbe(lambda state, t: torch.ones_like(state), EuclideanManifold())
    for size in (4, 4, 2):
        probe.start_batch()
        for step in (0.0, 0.5):
            state = torch.ones(size, 2, 4, 4)
            probe(state, torch.full((size,), step))

    assert probe.peak_angular is not None
    assert probe.peak_angular.shape[0] == 10


def test_probe_records_nothing_when_disabled() -> None:
    """A score arm has no angular velocity, so there is nothing to report."""
    from cyfm.metrics import AngularVelocityProbe

    probe = AngularVelocityProbe(
        lambda state, t: torch.ones_like(state), EuclideanManifold(), enabled=False
    )
    probe.start_batch()
    probe(torch.ones(2, 2, 4, 4), torch.zeros(2))
    assert probe.enabled is False
    assert probe.peak_angular is None


class _ZeroVelocity(torch.nn.Module):
    """Predicts no motion at all: the residual then equals the displacement."""

    def forward(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return torch.zeros(state.shape[0], 2, *state.shape[-2:])


def test_straightness_is_chunk_invariant_for_a_degenerate_model() -> None:
    """Chunking changes the draws, not the accumulation.

    A model that predicts zero explains none of the displacement, so straightness is
    1 whatever the chunk size. That pins the sums and means across chunks, which is
    the part chunking could get wrong; the draws themselves differ by construction.
    """
    from cyfm.evaluate import straightness
    from cyfm.manifolds import CylindricalManifold

    manifold = CylindricalManifold()
    device = torch.device("cpu")
    data = manifold.from_complex(torch.randn(8, 1, 8, 8, dtype=torch.complex64))

    whole = straightness(
        _ZeroVelocity(), manifold, data, device, torch.Generator().manual_seed(0), None, chunk=None
    )
    chunked = straightness(
        _ZeroVelocity(), manifold, data, device, torch.Generator().manual_seed(0), None, chunk=3
    )
    assert whole == pytest.approx(1.0)
    assert chunked == pytest.approx(1.0)


class TestAngularCapability:
    """The probe asks the geometry, instead of matching its name against a tuple.

    ``cyfm.evaluate`` held ``_ANGULAR_GEOMETRIES = ("euclidean", "cylindrical")``
    and a branch on the name to pick the formula. A geometry now states both for
    itself, so a new one cannot be left out of a table it belongs in by an
    omission from a tuple it has no reason to know about.
    """

    def test_the_score_arm_is_excluded_by_the_other_gate(self) -> None:
        """Defined for the representation, not reported for this arm.

        ``ComplexDiffusionManifold`` mixes in the flat representation and so has
        the induced formula; it still must not report it, because its network
        emits a score. The two gates are separate questions and this is the case
        that tells them apart.
        """
        from cyfm.manifolds.complex_diffusion import ComplexDiffusionManifold
        from cyfm.metrics import AngularVelocityProbe

        manifold = ComplexDiffusionManifold()
        assert manifold.reports_induced_angular_velocity is True
        assert manifold.predicts_velocity is False

        probe = AngularVelocityProbe(
            lambda state, t: torch.ones_like(state),
            manifold,
            enabled=manifold.predicts_velocity,
        )
        probe.start_batch()
        probe(torch.ones(2, 2, 4, 4), torch.zeros(2))
        assert probe.enabled is False
        assert probe.peak_angular is None

    def test_the_cylinder_reads_the_bounded_coordinate(self) -> None:
        """On the cylinder the angular velocity *is* a coordinate, bounded by pi."""
        from cyfm.manifolds.cylindrical import CylindricalManifold

        state = torch.rand(2, 3, 4, 4)
        velocity = torch.rand(2, 2, 4, 4)
        amplitude, angular = CylindricalManifold().induced_angular_velocity(state, velocity)
        assert torch.equal(angular, velocity[:, 1])
        assert torch.equal(amplitude, state[:, 0])

    def test_the_plane_induces_it_and_diverges_at_the_origin(self) -> None:
        """The divergence the probe exists to expose, as a number.

        A unit tangential velocity at amplitude A induces an angular velocity
        1/A, so halving the amplitude doubles it. That is Theorem 3's content.
        """
        from cyfm.manifolds.euclidean import EuclideanManifold

        manifold = EuclideanManifold()
        velocity = torch.tensor([[[[0.0]], [[1.0]]]])  # purely tangential at (A, 0)
        rates = []
        for radius in (1.0, 0.5, 0.25):
            state = torch.tensor([[[[radius]], [[0.0]]]])
            _, angular = manifold.induced_angular_velocity(state, velocity)
            rates.append(float(angular))
        assert rates == pytest.approx([1.0, 2.0, 4.0])

    def test_the_floor_stops_a_literal_division_by_zero(self) -> None:
        """Exactly at the origin the formula must return a number, not a nan."""
        from cyfm.manifolds.euclidean import EuclideanManifold

        amplitude, angular = EuclideanManifold().induced_angular_velocity(
            torch.zeros(1, 2, 2, 2), torch.ones(1, 2, 2, 2)
        )
        assert torch.isfinite(amplitude).all()
        assert torch.isfinite(angular).all()

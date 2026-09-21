"""The evaluation report, and the convention that governs what it omits.

``metrics.json`` omits the metric families an arm cannot report rather than
writing ``null`` or ``0.0``, and the table scripts rely on the absence. A reader
must not be able to mistake "not applicable" for "measured, and small". These
tests pin that, and pin the split between measuring and rendering: the report is
built from a payload, so an archived result can be re-rendered without
regenerating a sample.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

from cyfm.manifolds.euclidean import EuclideanManifold
from cyfm.pipelines.evaluation import (
    EvaluationPipeline,
    assert_training_domain,
    format_table,
    render_report,
    training_pipeline,
)


def _payload(*, with_path_metrics: bool) -> dict:
    row = {
        "num_steps": 1,
        "nfe": 1,
        "sliced_w2_complex": 0.1,
        "w2_amplitude": 0.2,
        "w2_phase_circular": 0.3,
        "dependence_gap": 0.4,
        "dependence_generated": 0.5,
        "dependence_reference": 0.6,
        "spatial_lag1_gap": 0.7,
        "spatial_lag1_generated": 0.8,
        "spatial_lag1_reference": 0.9,
        "radial_spectrum_gap": 1.0,
        "model_evaluations": 1.0,
        "executed_steps": 1.0,
    }
    payload: dict = {"manifold": "m", "sweep": [row]}
    if with_path_metrics:
        row |= {
            "peak_angular_velocity_median": 0.1,
            "peak_angular_velocity_max": 0.2,
            "peak_angular_velocity_near_t_half": 0.3,
            "min_amplitude_mean": 0.4,
            "min_amplitude_min": 0.5,
        }
        payload |= {"straightness": 0.9, "straightness_pairing": "ot"}
    return payload


class TestRenderReport:
    def test_a_velocity_arm_gets_both_extra_sections(self) -> None:
        report = render_report(_payload(with_path_metrics=True))
        assert "straightness (ot pairing): 0.90000" in report
        assert "ANGULAR VELOCITY ALONG THE PATH" in report

    def test_a_score_arm_gets_neither(self) -> None:
        """Not a zero row and not a null: the sections are absent.

        Straightness is the regression residual of a conditional *velocity*. A
        score arm's bridge returns -z/sigma, whose scale runs away as sigma
        falls, so the ratio would be dominated by the t near 1 end and would sit
        in the table looking comparable to a flow arm's path straightness.
        """
        report = render_report(_payload(with_path_metrics=False))
        assert "straightness" not in report
        assert "ANGULAR VELOCITY" not in report

    def test_the_headline_table_is_rendered_either_way(self) -> None:
        for path_metrics in (True, False):
            report = render_report(_payload(with_path_metrics=path_metrics))
            assert "GENERATIVE EVALUATION" in report
            assert "sliced_w2_complex" in report

    def test_rendering_needs_nothing_but_the_payload(self) -> None:
        """No model, no manifold, no checkpoint: an archive is enough.

        This is what the split buys. The network-free probes still fuse
        measurement with printing, which is why four of the paper's tables have
        no generator.
        """
        assert render_report(_payload(with_path_metrics=True))


class TestFormatTable:
    def test_columns_follow_the_requested_order(self) -> None:
        table = format_table([(1, {"b": 2.0, "a": 1.0})], ["a", "b"])
        header = table.splitlines()[0]
        assert header.index("a") < header.index("b")

    def test_one_row_per_step_count(self) -> None:
        table = format_table([(1, {"a": 1.0}), (4, {"a": 2.0})], ["a"])
        assert len(table.splitlines()) == 4  # header, rule, two rows


class TestTrainingDomain:
    def test_a_normalised_batch_reports_its_peak(self) -> None:
        field = torch.zeros(2, 1, 4, 4, dtype=torch.complex64)
        field[0, 0, 0, 0] = 1.0
        assert assert_training_domain(field) == pytest.approx(1.0)

    def test_a_field_cropped_below_one_is_accepted(self) -> None:
        """The bound is one-sided on purpose.

        Normalisation happens before the crop, so a field whose peak was cropped
        away is legitimately below one; only exceeding one proves the division
        never happened.
        """
        field = torch.full((1, 1, 2, 2), 0.3, dtype=torch.complex64)
        assert assert_training_domain(field) == pytest.approx(0.3)

    def test_a_raw_batch_is_rejected_and_says_why(self) -> None:
        """A raw batch must be rejected, not scored.

        Scoring against un-normalised fields measures the missing division, which
        once biased every absolute W2 in this module.
        """
        field = torch.full((1, 1, 2, 2), 5.0, dtype=torch.complex64)
        with pytest.raises(ValueError, match="not in the training domain"):
            assert_training_domain(field)


class TestTrainingPipelineTransform:
    def test_it_reproduces_the_representation_training_used(self) -> None:
        from cyfm.manifolds.cylindrical import CylindricalManifold

        manifold = CylindricalManifold()
        pipeline = training_pipeline({"crop_size": [16, 16]}, manifold)
        state = pipeline(torch.randn(1, 20, 20) + 1j * torch.randn(1, 20, 20))
        assert state.shape == (manifold.state_channels, 16, 16)

    def test_the_output_is_in_the_training_domain(self) -> None:
        """The whole point: a reference batch must survive assert_training_domain."""
        from cyfm.manifolds.euclidean import EuclideanManifold

        manifold = EuclideanManifold()
        pipeline = training_pipeline({"crop_size": [16, 16]}, manifold)
        state = pipeline(5.0 * (torch.randn(1, 20, 20) + 1j * torch.randn(1, 20, 20)))
        assert_training_domain(manifold.to_complex(state.unsqueeze(0)))


class TestEvaluationPipelineTeardown:
    def test_teardown_clears_references_and_cache(self) -> None:
        pipeline = EvaluationPipeline.__new__(EvaluationPipeline)
        pipeline.data_states = torch.zeros(2, 2, 4, 4)
        pipeline.reference = torch.zeros(2, 1, 4, 4, dtype=torch.complex64)

        pipeline.teardown()

        assert pipeline.data_states is None
        assert pipeline.reference is None

    def test_load_reference_calls_dataset_close(self, monkeypatch: pytest.MonkeyPatch) -> None:
        closed = False

        class ClosableDataset(Dataset):
            def __len__(self) -> int:
                return 4

            def __getitem__(self, idx: int) -> torch.Tensor:
                return torch.zeros(2, 16, 16)

            def close(self) -> None:
                nonlocal closed
                closed = True

        monkeypatch.setattr(
            "cyfm.pipelines.evaluation.build_dataset",
            lambda *args, **kwargs: ClosableDataset(),
        )

        pipeline = EvaluationPipeline.__new__(EvaluationPipeline)
        pipeline.config = type(  # type: ignore[assignment]
            "Config",
            (),
            {
                "evaluate": type(
                    "Eval",
                    (),
                    {"batch_size": 2, "num_workers": 0, "num_fields": 2},
                )()
            },
        )()
        pipeline.dataset_cfg = {}
        pipeline.manifold = EuclideanManifold()
        pipeline.device = torch.device("cpu")

        pipeline._load_reference()

        assert closed
        assert pipeline.data_states is not None
        assert pipeline.reference is not None

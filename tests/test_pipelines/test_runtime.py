"""Unit tests for the training loop's machinery.

None of this was reachable before. ``train.py`` was one Hydra ``main`` of 535
lines, so the only way to exercise the epoch reduction, the resume logic or the
checkpoint cadence was to run a training job. The suite had no test that ran one.
"""

from __future__ import annotations

import os

import pytest
import torch

from cyfm.config.schema import SchedulerConfig, TrainingConfig
from cyfm.core.pipeline import BasePipeline
from cyfm.pipelines.reporting import NullLogger, RunLogger, build_logger
from cyfm.pipelines.runtime import (
    CheckpointWriter,
    EpochAccumulator,
    StopController,
    build_optimizer_and_scheduler,
    maybe_resume,
)
from cyfm.utils.seeding import seed_everything

CPU = torch.device("cpu")


def _model() -> torch.nn.Module:
    return torch.nn.Linear(4, 4)


class TestBasePipeline:
    def test_phases_run_in_order(self) -> None:
        order: list[str] = []

        class Recorder(BasePipeline[str]):
            def setup(self) -> None:
                order.append("setup")

            def execute(self) -> str:
                order.append("execute")
                return "done"

            def teardown(self) -> None:
                order.append("teardown")

        assert Recorder().run() == "done"
        assert order == ["setup", "execute", "teardown"]

    def test_teardown_runs_when_execute_raises(self) -> None:
        """The guarantee the old entry points did not give.

        A failure mid-epoch left the W&B run open and the process group up,
        which on a cluster means the job's remaining ranks hang instead of
        exiting.
        """
        torn_down = []

        class Failing(BasePipeline[None]):
            def execute(self) -> None:
                raise RuntimeError("epoch blew up")

            def teardown(self) -> None:
                torn_down.append(True)

        with pytest.raises(RuntimeError, match="epoch blew up"):
            Failing().run()
        assert torn_down == [True]

    def test_setup_and_teardown_are_optional(self) -> None:
        class Minimal(BasePipeline[int]):
            def execute(self) -> int:
                return 7

        assert Minimal().run() == 7


class TestSeeding:
    def test_an_unseeded_run_gets_no_generator(self) -> None:
        assert seed_everything(None) is None

    def test_one_seed_gives_one_stream(self) -> None:
        first = seed_everything(3)
        assert first is not None
        a = torch.rand(4, generator=first)
        second = seed_everything(3)
        assert second is not None
        assert torch.equal(a, torch.rand(4, generator=second))

    def test_the_global_stream_is_seeded_too(self) -> None:
        """Weight initialisation reads the global RNG, not the loader's."""
        seed_everything(11)
        a = torch.rand(4)
        seed_everything(11)
        assert torch.equal(a, torch.rand(4))


class TestEpochAccumulator:
    def test_averages_over_batches(self) -> None:
        accumulator = EpochAccumulator()
        accumulator.add(2.0, {"amp": torch.tensor(1.0)})
        accumulator.add(4.0, {"amp": torch.tensor(3.0)})
        avg, components, batches = accumulator.reduce(CPU)
        assert (avg, batches) == (3.0, 2)
        assert components == {"amp": 2.0}

    def test_component_order_is_sorted_not_insertion(self) -> None:
        """The reduction is positional, so the ranks must agree on the order.

        Component names come from the manifold, whose dict order is not a
        contract. Getting this wrong does not raise; it silently attributes one
        geometry's amplitude loss to another's phase loss.
        """
        forward = EpochAccumulator()
        forward.add(1.0, {"amp": torch.tensor(1.0), "phi": torch.tensor(2.0)})
        reverse = EpochAccumulator()
        reverse.add(1.0, {"phi": torch.tensor(2.0), "amp": torch.tensor(1.0)})
        assert forward.reduce(CPU)[1] == reverse.reduce(CPU)[1]

    def test_an_empty_component_dict_is_fine(self) -> None:
        accumulator = EpochAccumulator()
        accumulator.add(1.0, {})
        assert accumulator.reduce(CPU) == (1.0, {}, 1)


class TestStopController:
    def test_a_touched_preempt_file_requests_a_stop(self, tmp_path) -> None:
        flag = tmp_path / "stop.now"
        controller = StopController(str(flag))
        assert controller.requested_locally() is False
        flag.write_text("")
        assert controller.requested_locally() is True

    def test_no_preempt_file_configured_is_not_a_stop(self) -> None:
        assert StopController(None).requested_locally() is False

    def test_should_stop_agrees_with_the_local_answer_off_ddp(self, tmp_path) -> None:
        flag = tmp_path / "stop.now"
        flag.write_text("")
        assert StopController(str(flag)).should_stop(CPU) is True


class TestScheduler:
    def test_an_unbuildable_scheduler_is_rejected(self) -> None:
        """Validated in two places on purpose, and the second is this one.

        SchedulerConfig rejects an unknown name; this asserts a name it accepts
        also has a branch here, so adding a member there without a builder fails
        loudly instead of silently training on cosine.
        """
        config = TrainingConfig()
        object.__setattr__(config.scheduler, "type", "StepLR")
        with pytest.raises(ValueError, match="No builder for"):
            build_optimizer_and_scheduler(_model(), config)

    def test_the_horizon_comes_from_the_epoch_count(self) -> None:
        _, scheduler = build_optimizer_and_scheduler(_model(), TrainingConfig(epochs=37))
        assert scheduler.T_max == 37

    def test_the_eta_min_reaches_the_schedule(self) -> None:
        config = TrainingConfig(scheduler=SchedulerConfig(eta_min=1e-3))
        _, scheduler = build_optimizer_and_scheduler(_model(), config)
        assert scheduler.eta_min == 1e-3


class TestResume:
    def test_disabled_starts_fresh(self, tmp_path) -> None:
        model = _model()
        optimizer, scheduler = build_optimizer_and_scheduler(model, TrainingConfig())
        state = maybe_resume(
            state_dir=str(tmp_path),
            device=CPU,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=10,
            enabled=False,
        )
        assert (state.start_epoch, state.run_id) == (0, None)

    def test_enabled_with_no_state_starts_fresh(self, tmp_path) -> None:
        model = _model()
        optimizer, scheduler = build_optimizer_and_scheduler(model, TrainingConfig())
        state = maybe_resume(
            state_dir=str(tmp_path),
            device=CPU,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=10,
            enabled=True,
        )
        assert state.start_epoch == 0

    def test_a_longer_run_reanneals_on_the_new_horizon(self, tmp_path) -> None:
        """T_max travels inside the scheduler state.

        Extending training.epochs across a resume would otherwise keep annealing
        on the old horizon and send the learning rate back up the far side of the
        cosine.
        """
        model = _model()
        optimizer, scheduler = build_optimizer_and_scheduler(model, TrainingConfig(epochs=10))
        writer = CheckpointWriter(str(tmp_path / "state"), str(tmp_path / "ckpt"))
        writer.write(
            epoch=4,
            total_epochs=10,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            run_id=None,
        )

        fresh = _model()
        new_optimizer, new_scheduler = build_optimizer_and_scheduler(
            fresh, TrainingConfig(epochs=40)
        )
        state = maybe_resume(
            state_dir=str(tmp_path / "state"),
            device=CPU,
            model=fresh,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            epochs=40,
            enabled=True,
        )
        assert state.start_epoch == 5
        assert new_scheduler.T_max == 40


class TestCheckpointWriter:
    def _write(self, tmp_path, epoch: int, total: int = 20) -> CheckpointWriter:
        model = _model()
        optimizer, scheduler = build_optimizer_and_scheduler(model, TrainingConfig())
        writer = CheckpointWriter(str(tmp_path / "state"), str(tmp_path / "ckpt"))
        writer.write(
            epoch=epoch,
            total_epochs=total,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            run_id=None,
        )
        return writer

    def test_the_resume_state_is_written_every_epoch(self, tmp_path) -> None:
        """Its staleness is exactly what an interruption costs in GPU time."""
        self._write(tmp_path, epoch=0)
        assert (tmp_path / "state" / "last.pt").exists()

    def test_an_inference_checkpoint_is_not(self, tmp_path) -> None:
        self._write(tmp_path, epoch=0)
        assert list((tmp_path / "ckpt").glob("*.pt")) == []

    def test_the_interval_writes_one(self, tmp_path) -> None:
        self._write(tmp_path, epoch=9)
        assert (tmp_path / "ckpt" / "checkpoint_epoch_10.pt").exists()

    def test_the_final_epoch_always_writes_one(self, tmp_path) -> None:
        """Otherwise a run whose length is not a multiple of the interval ends
        with no checkpoint for evaluate.py to find."""
        self._write(tmp_path, epoch=6, total=7)
        assert (tmp_path / "ckpt" / "checkpoint_epoch_7.pt").exists()

    def test_the_two_files_are_not_interchangeable(self, tmp_path) -> None:
        """resolve_checkpoint globs for the newest .pt and would choke on last.pt.

        The resume state is a payload dict; an inference checkpoint is a bare
        state dict. They live in different trees for that reason.
        """
        self._write(tmp_path, epoch=9)
        resume = torch.load(tmp_path / "state" / "last.pt", weights_only=False)
        inference = torch.load(tmp_path / "ckpt" / "checkpoint_epoch_10.pt", weights_only=True)
        assert set(resume) >= {"model", "optimizer", "scheduler", "epochs_completed"}
        assert set(inference) == {"weight", "bias"}

    def test_a_non_main_rank_writes_nothing(self, tmp_path) -> None:
        model = _model()
        optimizer, scheduler = build_optimizer_and_scheduler(model, TrainingConfig())
        writer = CheckpointWriter(str(tmp_path / "state"), str(tmp_path / "ckpt"), enabled=False)
        writer.write(
            epoch=9,
            total_epochs=10,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            run_id=None,
        )
        assert not os.path.exists(tmp_path / "state")
        assert not os.path.exists(tmp_path / "ckpt")


class TestLogger:
    def test_a_local_run_gets_the_null_logger(self) -> None:
        """conf/logging/default.yaml ships use_wandb: false, so this is the
        shipped behaviour rather than a degraded fallback."""
        logger = build_logger(
            use_wandb=False,
            is_main=True,
            project="p",
            name="n",
            directory=".",
            config={},
        )
        assert isinstance(logger, NullLogger)

    def test_a_non_main_rank_never_logs_remotely(self) -> None:
        """N ranks logging the same step would interleave into N runs."""
        logger = build_logger(
            use_wandb=True,
            is_main=False,
            project="p",
            name="n",
            directory=".",
            config={},
        )
        assert isinstance(logger, NullLogger)

    def test_the_null_logger_satisfies_the_protocol(self) -> None:
        logger: RunLogger = NullLogger()
        logger.log({"loss": 1.0})
        logger.log_image("k", torch.zeros(2, 2), "caption")
        logger.finish()
        assert logger.run_id is None

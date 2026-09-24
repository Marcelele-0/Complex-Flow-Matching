"""Every default in the schema must be the default in ``conf/``.

This is the test that makes :mod:`cyfm.config.schema` honest. Nothing stops a
Python dataclass default from drifting away from the YAML it mirrors, and five of
them already had: the previous code read ``matmul_precision`` as ``"highest"``
where the config says ``"high"``, named the W&B project differently from the config,
and -- the one that could cost a run -- defaulted ``paths.state_dir`` to a single
unkeyed ``outputs/state`` where the config keys it by experiment name, so two
concurrent runs would overwrite each other's resume state.

Reading the YAML here rather than restating it is the point: if ``conf/`` moves,
this fails.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from cyfm.config.schema import (
    EvaluateConfig,
    LoggingConfig,
    PathsConfig,
    RunConfig,
    SchedulerConfig,
    TrainingConfig,
)

CONF = Path(__file__).resolve().parents[2] / "conf"


def _yaml(group: str) -> dict[str, Any]:
    """The group as *Hydra* reads it, which is not what every YAML parser reads.

    OmegaConf is more permissive than YAML 1.1 about scientific notation:
    ``2e-4`` is a float to it and a string to ``yaml.safe_load``, which has no
    decimal point to key on. Comparing against PyYAML would have this test
    assert that ``learning_rate`` is the string ``"2e-4"``, which is true of
    nothing the code ever sees.
    """
    return dict(OmegaConf.to_container(OmegaConf.load(CONF / group), resolve=False))


class TestDefaultsMirrorTheYaml:
    @pytest.mark.parametrize(
        "key",
        [
            "epochs",
            "batch_size",
            "num_workers",
            "learning_rate",
            "weight_decay",
            "matmul_precision",
            "bridge",
            "coupling",
            "auto_resume",
            "preempt_file",
            "seed",
            "grad_clip",
            "compile",
        ],
    )
    def test_training(self, key: str) -> None:
        assert getattr(TrainingConfig(), key) == _yaml("training/default.yaml")[key]

    @pytest.mark.parametrize("key", ["type", "eta_min"])
    def test_scheduler(self, key: str) -> None:
        assert getattr(SchedulerConfig(), key) == _yaml("training/default.yaml")["scheduler"][key]

    @pytest.mark.parametrize("key", ["use_wandb", "project_name", "experiment_name"])
    def test_logging(self, key: str) -> None:
        assert getattr(LoggingConfig(), key) == _yaml("logging/default.yaml")[key]

    @pytest.mark.parametrize(
        "key",
        [
            "num_fields",
            "batch_size",
            "num_workers",
            "straightness_batch_size",
            "num_projections",
            "seed",
            "run_name",
            "checkpoint_path",
        ],
    )
    def test_evaluate(self, key: str) -> None:
        assert getattr(EvaluateConfig(), key) == _yaml("evaluate/default.yaml")[key]

    def test_evaluate_nfe(self) -> None:
        assert list(EvaluateConfig().nfe) == _yaml("evaluate/default.yaml")["nfe"]

    def test_the_training_group_has_no_key_the_schema_ignores(self) -> None:
        """A key added to the YAML must be added here, or it is decorative.

        ``training.scheduler.type`` was exactly that: documented in the config,
        never read, with CosineAnnealingLR hardcoded in the loop.
        """
        yaml_keys = set(_yaml("training/default.yaml"))
        schema_keys = set(vars(TrainingConfig()))
        assert yaml_keys - schema_keys == set()


class TestInterpolatedPathsAreDerived:
    """``conf/paths`` writes two of the three as interpolations.

    A Python default cannot be an interpolation, so these are derived from the
    same inputs instead of being given some unrelated literal -- which is how
    ``state_dir`` came to differ.
    """

    def test_state_dir_is_keyed_by_the_experiment_name(self) -> None:
        paths = PathsConfig.from_config({}, experiment_name="my_run")
        assert paths.state_dir == "outputs/state/my_run"

    def test_state_dir_matches_the_configs_interpolation(self) -> None:
        template = _yaml("paths/default.yaml")["state_dir"]
        assert template == "outputs/state/${logging.experiment_name}"
        expected = template.replace("${logging.experiment_name}", "some_run")
        assert PathsConfig.from_config({}, experiment_name="some_run").state_dir == expected

    def test_checkpoint_dir_follows_the_output_dir(self) -> None:
        template = _yaml("paths/default.yaml")["checkpoint_dir"]
        assert template == "${paths.output_dir}/checkpoints"
        paths = PathsConfig.from_config({"output_dir": "/tmp/run"})
        assert paths.checkpoint_dir == "/tmp/run/checkpoints"

    def test_two_experiment_names_never_share_a_resume_state(self) -> None:
        """The hazard the key exists to prevent, stated as a test.

        Under the old unkeyed default two concurrent runs wrote one ``last.pt``,
        and each would have resumed from the other's optimizer moments.
        """
        first = RunConfig.from_config({"logging": {"experiment_name": "arm_a"}})
        second = RunConfig.from_config({"logging": {"experiment_name": "arm_b"}})
        assert first.paths.state_dir != second.paths.state_dir


class TestValidation:
    def test_an_unknown_scheduler_is_rejected_rather_than_ignored(self) -> None:
        with pytest.raises(ValueError, match="training.scheduler.type"):
            SchedulerConfig(type="StepLR")

    def test_the_shipped_scheduler_is_accepted(self) -> None:
        built = SchedulerConfig.from_config({"type": "CosineAnnealingLR"})
        assert built.type == "CosineAnnealingLR"


class TestParsedOnce:
    def test_the_whole_tree_yields_four_sections(self) -> None:
        run = RunConfig.from_config(
            {
                "training": {"epochs": 3, "seed": None, "grad_clip": None},
                "logging": {"experiment_name": "x"},
                "paths": {"output_dir": "/out"},
                "evaluate": {"nfe": [1, 2]},
            }
        )
        assert run.training.epochs == 3
        assert run.training.seed is None
        assert run.training.grad_clip is None
        assert run.paths.checkpoint_dir == "/out/checkpoints"
        assert run.paths.state_dir == "outputs/state/x"
        assert list(run.evaluate.nfe) == [1, 2]

    def test_an_empty_tree_is_all_defaults(self) -> None:
        assert RunConfig.from_config({}) == RunConfig(
            training=TrainingConfig(),
            logging=LoggingConfig(),
            paths=PathsConfig(),
            evaluate=EvaluateConfig(),
        )

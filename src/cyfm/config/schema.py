"""The config sections the entry points read, parsed once into typed records.

Two problems this replaces.

**The config was re-fetched at every use.** ``train.py`` evaluated
``cfg.get("training", {})`` fourteen times, ``cfg.get("dataset", {})`` four and
``cfg.get("paths", {})`` three, each with its own inline default. Nothing checked
that the fourteen agreed, and nothing could: they were fourteen independent
expressions.

**The inline defaults disagreed with ``conf/``.** Five of them, so a caller who
built a pipeline without Hydra got different behaviour from a Hydra run with the
same settings unset. The worst was ``paths.state_dir``: the code said
``outputs/state`` while the config says ``outputs/state/${logging.experiment_name}``,
and that key exists precisely so two runs do not fight over one ``last.pt`` --
which is what two unkeyed runs would have done.

The defaults below therefore reproduce ``conf/`` exactly, including the two that
are interpolations there and are derived here from the same inputs.
``tests/test_config/test_schema_matches_conf.py`` reads the YAML and asserts
field by field that they still do, so a divergence is a failing test rather than
a behaviour difference nobody sees.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Self

from cyfm.config.resolve import as_plain_dict

__all__ = [
    "SCHEDULERS",
    "EvaluateConfig",
    "LoggingConfig",
    "PathsConfig",
    "RunConfig",
    "SchedulerConfig",
    "TrainingConfig",
]

# Learning-rate schedules this loop can build. One entry, and it is the one every
# paper run used; the point of the tuple is that `training.scheduler.type` is
# checked against it rather than ignored. It was ignored: CosineAnnealingLR was
# hardcoded, so setting any other value silently trained on cosine anyway.
SCHEDULERS = ("CosineAnnealingLR",)


@dataclass(frozen=True)
class SchedulerConfig:
    """``training.scheduler``."""

    type: str = "CosineAnnealingLR"
    eta_min: float = 1e-6

    def __post_init__(self) -> None:
        if self.type not in SCHEDULERS:
            raise ValueError(
                f"training.scheduler.type must be one of {list(SCHEDULERS)}, got {self.type!r}."
            )

    @classmethod
    def from_config(cls, section: Mapping[str, Any] | None) -> Self:
        """Build from the ``training.scheduler`` group."""
        values = as_plain_dict(section)
        return cls(
            type=str(values.get("type", "CosineAnnealingLR")),
            eta_min=float(values.get("eta_min", 1e-6)),
        )


@dataclass(frozen=True)
class TrainingConfig:
    """``training``: everything the loop needs that is not geometry or data."""

    epochs: int = 100
    batch_size: int = 4
    num_workers: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    matmul_precision: str = "high"
    bridge: str = "noise"
    coupling: str = "independent"
    auto_resume: bool = False
    preempt_file: str | None = None
    seed: int | None = 0
    grad_clip: float | None = 1.0
    compile: bool = True
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, section: Mapping[str, Any] | None) -> Self:
        """Build from the ``training`` group."""
        values = as_plain_dict(section)
        grad_clip = values.get("grad_clip", 1.0)
        seed = values.get("seed", 0)
        preempt_file = values.get("preempt_file")
        return cls(
            epochs=int(values.get("epochs", 100)),
            batch_size=int(values.get("batch_size", 4)),
            num_workers=int(values.get("num_workers", 4)),
            learning_rate=float(values.get("learning_rate", 2e-4)),
            weight_decay=float(values.get("weight_decay", 1e-4)),
            matmul_precision=str(values.get("matmul_precision", "high")),
            bridge=str(values.get("bridge", "noise")),
            coupling=str(values.get("coupling", "independent")),
            auto_resume=bool(values.get("auto_resume", False)),
            preempt_file=None if preempt_file is None else str(preempt_file),
            seed=None if seed is None else int(seed),
            grad_clip=None if grad_clip is None else float(grad_clip),
            compile=bool(values.get("compile", True)),
            scheduler=SchedulerConfig.from_config(values.get("scheduler")),
            loss=as_plain_dict(values.get("loss")),
        )


@dataclass(frozen=True)
class LoggingConfig:
    """``logging``. ``experiment_name`` is this repository's run identity: it names
    the Hydra run directory, the W&B run, the resume state and evaluate's
    ``run_name``."""

    use_wandb: bool = False
    project_name: str = "complex_flow_matching"
    experiment_name: str = "cyfm_run"

    @classmethod
    def from_config(cls, section: Mapping[str, Any] | None) -> Self:
        """Build from the ``logging`` group."""
        values = as_plain_dict(section)
        return cls(
            use_wandb=bool(values.get("use_wandb", False)),
            project_name=str(values.get("project_name", "complex_flow_matching")),
            experiment_name=str(values.get("experiment_name", "cyfm_run")),
        )


@dataclass(frozen=True)
class PathsConfig:
    """``paths``. Two of the three are interpolations in the config, and are
    derived here from the same inputs rather than given an unrelated literal."""

    output_dir: str = "."
    checkpoint_dir: str = "./checkpoints"
    state_dir: str = "outputs/state/cyfm_run"

    @classmethod
    def from_config(
        cls, section: Mapping[str, Any] | None, experiment_name: str = "cyfm_run"
    ) -> Self:
        """Build from the ``paths`` group.

        Args:
            section: The ``paths`` group.
            experiment_name: ``logging.experiment_name``, which the config
                interpolates into ``state_dir``. Passed in rather than defaulted
                because getting it wrong means two runs share one ``last.pt``.

        Returns:
            The resolved paths.
        """
        values = as_plain_dict(section)
        output_dir = str(values.get("output_dir", "."))
        return cls(
            output_dir=output_dir,
            checkpoint_dir=str(values.get("checkpoint_dir", f"{output_dir}/checkpoints")),
            state_dir=str(values.get("state_dir", f"outputs/state/{experiment_name}")),
        )


@dataclass(frozen=True)
class EvaluateConfig:
    """``evaluate``: the generative sweep's cohort, budget and seed."""

    num_fields: int = 64
    batch_size: int = 16
    num_workers: int = 0
    straightness_batch_size: int | None = None
    nfe: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 100)
    num_projections: int = 256
    seed: int = 0
    run_name: str | None = None
    checkpoint_path: str | None = None

    @classmethod
    def from_config(cls, section: Mapping[str, Any] | None) -> Self:
        """Build from the ``evaluate`` group."""
        values = as_plain_dict(section)
        straightness = values.get("straightness_batch_size")
        run_name = values.get("run_name")
        checkpoint_path = values.get("checkpoint_path")
        return cls(
            num_fields=int(values.get("num_fields", 64)),
            batch_size=int(values.get("batch_size", 16)),
            num_workers=int(values.get("num_workers", 0)),
            straightness_batch_size=None if straightness is None else int(straightness),
            nfe=tuple(int(n) for n in values.get("nfe", (1, 2, 4, 8, 16, 32, 64, 100))),
            num_projections=int(values.get("num_projections", 256)),
            seed=int(values.get("seed", 0)),
            run_name=None if run_name is None else str(run_name),
            checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
        )


@dataclass(frozen=True)
class RunConfig:
    """Every section an entry point reads, parsed once.

    Order matters in :meth:`from_config`: ``paths.state_dir`` interpolates
    ``logging.experiment_name`` in the config, so the logging section is parsed
    first and handed to the paths section.
    """

    training: TrainingConfig
    logging: LoggingConfig
    paths: PathsConfig
    evaluate: EvaluateConfig

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> Self:
        """Parse a composed config tree.

        Args:
            cfg: The whole config, as a ``DictConfig`` or a plain mapping.

        Returns:
            The four typed sections.
        """
        logging_cfg = LoggingConfig.from_config(cfg.get("logging"))
        return cls(
            training=TrainingConfig.from_config(cfg.get("training")),
            logging=logging_cfg,
            paths=PathsConfig.from_config(
                cfg.get("paths"), experiment_name=logging_cfg.experiment_name
            ),
            evaluate=EvaluateConfig.from_config(cfg.get("evaluate")),
        )

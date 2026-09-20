"""Where a run's numbers go, behind one interface.

``train.py`` carried a module-level ``HAS_WANDB`` flag set by a ``try: import
wandb``, a ``use_wandb`` local recomputed from it and the rank, and four separate
``if use_wandb:`` blocks inside the loop -- one per logging site, each also
responsible for not logging. The loop therefore knew about Weights & Biases,
about ImportError, and about which rank it was on.

It needs to know none of that. A run holds a :class:`RunLogger`; the null
implementation is the one a local run gets, and it is not a degraded mode but the
default the shipped ``conf/logging/default.yaml`` selects.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch

__all__ = ["NullLogger", "RunLogger", "WandbLogger", "build_logger"]


class RunLogger(Protocol):
    """Somewhere to send scalars, an occasional image, and a closing signal."""

    def log(self, values: dict[str, Any]) -> None:
        """Record one step's or one epoch's scalars."""
        ...

    def log_image(self, key: str, image: torch.Tensor, caption: str) -> None:
        """Record one image, already reduced to a 2D tensor."""
        ...

    def finish(self) -> None:
        """Close the run. Safe to call when nothing was ever logged."""
        ...

    @property
    def run_id(self) -> str | None:
        """Identifier a requeued job reattaches to, or ``None``."""
        ...


class NullLogger:
    """Discards everything. What a local run gets, and what tests get.

    Not a fallback for a failed import: ``conf/logging/default.yaml`` ships with
    ``use_wandb: false``, so this is the shipped behaviour and stdout is the
    record.
    """

    def log(self, values: dict[str, Any]) -> None:
        """Discard the scalars."""

    def log_image(self, key: str, image: torch.Tensor, caption: str) -> None:
        """Discard the image."""

    def finish(self) -> None:
        """Nothing to close."""

    @property
    def run_id(self) -> str | None:
        """Always ``None``: there is no remote run to reattach to."""
        return None


class WandbLogger:
    """Weights & Biases, initialised eagerly so a broken setup fails at setup.

    Args:
        project: W&B project name.
        name: Run name; this repository uses ``logging.experiment_name``
            throughout, which also names the Hydra run directory and the resume
            state.
        directory: Where W&B writes its local cache.
        config: The resolved config, recorded with the run.
        resume_id: Run to reattach to, from a resume state. A requeued job
            continues the run it was writing, so one training curve is one line
            rather than one line per scheduler decision.

    Raises:
        ImportError: If ``wandb`` is not installed. Raised rather than silently
            downgrading to :class:`NullLogger`: a run asked for W&B and would
            otherwise finish with its metrics nowhere.
    """

    def __init__(
        self,
        project: str,
        name: str,
        directory: str,
        config: dict[str, Any],
        resume_id: str | None = None,
    ) -> None:
        import wandb

        self._wandb = wandb
        wandb.init(
            project=project,
            name=name,
            dir=directory,
            config=config,
            id=resume_id,
            resume="allow",
        )

    def log(self, values: dict[str, Any]) -> None:
        """Send the scalars to W&B."""
        self._wandb.log(values)

    def log_image(self, key: str, image: torch.Tensor, caption: str) -> None:
        """Send one image to W&B."""
        self._wandb.log({key: self._wandb.Image(image.numpy(), caption=caption)})

    def finish(self) -> None:
        """Close the W&B run."""
        self._wandb.finish()

    @property
    def run_id(self) -> str | None:
        """The W&B run id, saved into the resume state."""
        run = self._wandb.run
        return None if run is None else str(run.id)


def build_logger(
    *,
    use_wandb: bool,
    is_main: bool,
    project: str,
    name: str,
    directory: str,
    config: dict[str, Any],
    resume_id: str | None = None,
) -> RunLogger:
    """Pick the logger this rank should use.

    Args:
        use_wandb: ``logging.use_wandb``.
        is_main: Whether this is rank 0. Only rank 0 logs remotely: N ranks
            logging the same step would interleave into N runs.
        project: W&B project name.
        name: Run name.
        directory: Local cache directory.
        config: The resolved config, recorded with the run.
        resume_id: Run to reattach to.

    Returns:
        A :class:`WandbLogger` on rank 0 when asked for, else a
        :class:`NullLogger`.
    """
    if not (use_wandb and is_main):
        return NullLogger()
    return WandbLogger(
        project=project, name=name, directory=directory, config=config, resume_id=resume_id
    )

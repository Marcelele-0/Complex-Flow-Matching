"""Resumable training state for jobs the scheduler is allowed to interrupt.

A cluster run ends when the wall clock or the preemption policy says so, not when
the loop finishes, so the periodic weight-only checkpoints ``train.py`` writes for
inference are not enough to restart one. The optimizer moments and the cosine
schedule position have to come back too - resuming from weights alone silently
restarts the learning rate at its initial value, which is a different experiment.

The write is atomic because the interruption and the save race each other. A
half-written ``last.pt`` is worse than no file at all: the requeued job would fail
to load it *after* the state it needed had already been overwritten.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from cfm.utils.distributed import unwrap_model

STATE_FILENAME = "last.pt"


def state_path(state_dir: str) -> str:
    """Path of the resume file inside ``state_dir``."""
    return os.path.join(state_dir, STATE_FILENAME)


def save_training_state(
    state_dir: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epochs_completed: int,
    wandb_run_id: str | None = None,
) -> str:
    """Write everything needed to continue this run, atomically.

    Args:
        state_dir: Directory for the resume file. Created if missing. Must be stable
            across restarts, which the Hydra run directory is not - it carries a
            timestamp, so a requeued job would look for the file in a directory that
            has never been written to.
        model: The training model, wrapped or not; only the bare module is stored.
        optimizer: Optimizer whose moments continue the run.
        scheduler: LR scheduler whose position continues the run.
        epochs_completed: Number of epochs fully finished, i.e. the epoch index the
            resumed run starts from.
        wandb_run_id: W&B run to reattach to, so a requeue continues one curve
            instead of starting a second run beside it.

    Returns:
        Path of the written file.
    """
    os.makedirs(state_dir, exist_ok=True)
    payload = {
        "epochs_completed": epochs_completed,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "wandb_run_id": wandb_run_id,
    }

    final_path = state_path(state_dir)
    tmp_path = f"{final_path}.tmp"
    with open(tmp_path, "wb") as handle:
        torch.save(payload, handle)
        # torch.save leaves the bytes in the page cache. fsync before the rename is
        # what makes the file survive the node going away mid-save.
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, final_path)
    return final_path


def load_training_state(state_dir: str, device: torch.device) -> dict[str, Any] | None:
    """Read the resume file from ``state_dir``, or ``None`` if the run is new.

    Args:
        state_dir: Directory searched for :data:`STATE_FILENAME`.
        device: Device the stored tensors are mapped onto.

    Returns:
        The payload written by :func:`save_training_state`, or ``None``.
    """
    path = state_path(state_dir)
    if not os.path.isfile(path):
        return None
    payload: dict[str, Any] = torch.load(path, map_location=device, weights_only=True)
    return payload

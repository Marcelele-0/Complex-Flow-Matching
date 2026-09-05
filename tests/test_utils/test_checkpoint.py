"""Resume state has to survive being written while the node is going away.

These cover the two ways a resume silently goes wrong: a payload that restores the
weights but not the schedule, and a file that was half-written when the job died.
"""

from __future__ import annotations

import os

import pytest
import torch
from torch import nn

from cfm.utils.checkpoint import (
    STATE_FILENAME,
    load_training_state,
    save_training_state,
    state_path,
)

CPU = torch.device("cpu")


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def build_run(lr: float = 0.1, epochs: int = 10):
    model = TinyNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return model, optimizer, scheduler


def step_once(model, optimizer) -> None:
    """Give the optimizer real moments, so a dropped state would show up."""
    loss = model(torch.ones(2, 4)).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def test_load_returns_none_for_a_fresh_run(tmp_path) -> None:
    assert load_training_state(str(tmp_path), CPU) is None


def test_save_then_load_round_trips_every_component(tmp_path) -> None:
    model, optimizer, scheduler = build_run()
    step_once(model, optimizer)
    scheduler.step()

    save_training_state(
        str(tmp_path),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs_completed=7,
        wandb_run_id="abc123",
    )
    payload = load_training_state(str(tmp_path), CPU)

    assert payload is not None
    assert payload["epochs_completed"] == 7
    assert payload["wandb_run_id"] == "abc123"
    assert set(payload["model"]) == set(model.state_dict())
    # Non-empty optimizer state is the part that distinguishes a real resume from
    # reloading weights into a fresh AdamW.
    assert payload["optimizer"]["state"]
    assert payload["scheduler"]["last_epoch"] == 1


def test_restored_run_continues_rather_than_restarts(tmp_path) -> None:
    model, optimizer, scheduler = build_run(epochs=10)
    for _ in range(4):
        step_once(model, optimizer)
        scheduler.step()
    expected_lr = optimizer.param_groups[0]["lr"]

    save_training_state(
        str(tmp_path),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs_completed=4,
    )

    fresh_model, fresh_optimizer, fresh_scheduler = build_run(epochs=10)
    assert fresh_optimizer.param_groups[0]["lr"] != pytest.approx(expected_lr)

    payload = load_training_state(str(tmp_path), CPU)
    assert payload is not None
    fresh_model.load_state_dict(payload["model"])
    fresh_optimizer.load_state_dict(payload["optimizer"])
    fresh_scheduler.load_state_dict(payload["scheduler"])

    assert fresh_optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)
    for restored, original in zip(
        fresh_model.state_dict().values(), model.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(restored, original)


def test_wandb_run_id_defaults_to_none(tmp_path) -> None:
    model, optimizer, scheduler = build_run()
    save_training_state(
        str(tmp_path),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs_completed=1,
    )

    payload = load_training_state(str(tmp_path), CPU)
    assert payload is not None
    assert payload["wandb_run_id"] is None


def test_save_creates_the_directory(tmp_path) -> None:
    state_dir = tmp_path / "does" / "not" / "exist"
    model, optimizer, scheduler = build_run()

    written = save_training_state(
        str(state_dir),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs_completed=1,
    )

    assert written == state_path(str(state_dir))
    assert os.path.isfile(written)


def test_save_leaves_no_temporary_file_behind(tmp_path) -> None:
    model, optimizer, scheduler = build_run()
    save_training_state(
        str(tmp_path),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs_completed=1,
    )

    assert sorted(p.name for p in tmp_path.iterdir()) == [STATE_FILENAME]


def test_overwrite_never_exposes_a_partial_file(tmp_path) -> None:
    """The previous state must stay loadable until the new one is complete.

    A job killed during a save would otherwise come back to a truncated file, which
    is worse than coming back to the epoch before it.
    """
    model, optimizer, scheduler = build_run()
    save_training_state(
        str(tmp_path),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs_completed=1,
    )

    original_bytes = (tmp_path / STATE_FILENAME).read_bytes()

    # Fail the save at the rename, after the temporary file is fully written.
    def explode(*_args, **_kwargs):
        raise OSError("node went away")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "replace", explode)
        with pytest.raises(OSError):
            save_training_state(
                str(tmp_path),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs_completed=99,
            )

    assert (tmp_path / STATE_FILENAME).read_bytes() == original_bytes
    payload = load_training_state(str(tmp_path), CPU)
    assert payload is not None
    assert payload["epochs_completed"] == 1

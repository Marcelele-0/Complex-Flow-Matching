"""The pieces a training loop needs around the five lines that are the algorithm.

Everything here was inline in ``train.py``'s ``main``. None of it is specific to
a geometry, and none of it was reachable from a test.

The accumulator and the stop controller carry invariants worth naming, because
both are collective: every rank must agree, or the ranks deadlock. The reduction
is positional, so the component names are sorted before it; and a rank that
stopped on a signal its peers never saw would leave them blocked in the next
epoch's collectives, so the decision is all-reduced too.
"""

from __future__ import annotations

import os
import signal
from types import FrameType
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from cyfm.config.schema import TrainingConfig
from cyfm.utils.checkpoint import load_training_state, save_training_state, state_path
from cyfm.utils.distributed import (
    DistributedContext,
    any_across_ranks,
    print_main,
    sum_across_ranks,
    unwrap_model,
)

__all__ = [
    "CheckpointWriter",
    "EpochAccumulator",
    "ResumeState",
    "StopController",
    "build_dataloader",
    "build_optimizer_and_scheduler",
    "maybe_resume",
    "wrap_for_execution",
]

# Set by the handler, read at the end of every epoch. Module level because a
# signal handler cannot take a closure argument; StopController owns the reading.
_STOP_REQUESTED = False


def _request_stop(signum: int, _frame: FrameType | None) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print_main(f"\nSignal {signal.Signals(signum).name} received: stopping after this epoch.")


class StopController:
    """Decides, collectively, whether the run should finish after this epoch.

    Two channels, because on a cluster a signal has to survive being relayed from
    the batch script through ``srun``, ``uv`` and the ``torchrun`` agent before it
    reaches this process, and it does not: the agent has no SIGUSR1 handler, so
    the workers are killed outright instead of being allowed to finish the epoch.
    The batch script therefore touches a file no launcher can swallow, and the
    signal handler stays for the runs with no batch script in the way.

    Args:
        preempt_file: Path the batch script touches, or ``None`` outside one.
    """

    def __init__(self, preempt_file: str | None = None) -> None:
        self.preempt_file = preempt_file

    def install(self) -> None:
        """Catch the two signals a scheduler sends.

        SIGTERM is what ``scancel`` and most preemption policies send; SIGUSR1 is
        the convention for a grace warning.
        """
        signal.signal(signal.SIGUSR1, _request_stop)
        signal.signal(signal.SIGTERM, _request_stop)

    def requested_locally(self) -> bool:
        """Whether *this* rank has been asked to stop."""
        if _STOP_REQUESTED:
            return True
        return self.preempt_file is not None and os.path.exists(self.preempt_file)

    def should_stop(self, device: torch.device) -> bool:
        """Whether *any* rank has been asked to stop.

        All-reduced on purpose: a rank stopping on a signal its peers never
        received would leave them blocked in the next epoch's collectives.

        Args:
            device: Device the collective runs on.

        Returns:
            True if any rank asked to stop.
        """
        return bool(any_across_ranks(self.requested_locally(), device))


class EpochAccumulator:
    """Sums the epoch's losses, then reduces them across ranks in one collective.

    The reduction is *positional*: the values are packed into one list and
    summed, so every rank has to pack them in the same order. Component names
    come from the manifold, whose dict order is not a contract, so they are
    sorted. Getting that wrong does not raise -- it silently attributes one
    geometry's amplitude loss to another's phase loss.
    """

    def __init__(self) -> None:
        self.total = 0.0
        self.components: dict[str, float] = {}
        self.batches = 0

    def add(self, loss: float, components: dict[str, torch.Tensor]) -> None:
        """Accumulate one batch.

        Args:
            loss: The scalar total for this batch.
            components: The manifold's own breakdown of it.
        """
        self.total += loss
        for name, value in components.items():
            self.components[name] = self.components.get(name, 0.0) + value.item()
        self.batches += 1

    def reduce(self, device: torch.device) -> tuple[float, dict[str, float], int]:
        """Average over every rank's batches, in one collective.

        Args:
            device: Device the collective runs on.

        Returns:
            ``(average_loss, average_components, total_batches)``. The average is
            over the whole epoch rather than over this rank's shard.
        """
        names = sorted(self.components)
        totals = sum_across_ranks(
            [self.total, *(self.components[name] for name in names), float(self.batches)],
            device,
        )
        batches = totals[-1]
        return (
            totals[0] / batches,
            {name: total / batches for name, total in zip(names, totals[1:-1], strict=True)},
            int(batches),
        )


class CheckpointWriter:
    """Two cadences, for two different readers.

    The resume state is written every epoch, whatever the inference interval is:
    it is the file a requeued job comes back from, so its staleness is exactly
    the amount of GPU time an interruption costs. The inference checkpoint is a
    bare ``state_dict`` written every ``interval`` epochs and at the end, with
    ``torch.compile``'s ``_orig_mod.`` prefix already stripped, because that is
    what ``evaluate.py`` globs for.

    They are deliberately not the same file, and not in the same tree:
    ``resolve_checkpoint`` globs ``outputs/train/`` for the newest ``.pt``, and
    the resume state is a payload dict rather than a bare state dict, so it would
    be found and would fail to load.

    Args:
        state_dir: Where ``last.pt`` goes.
        checkpoint_dir: Where the inference checkpoints go.
        interval: Epochs between inference checkpoints.
        enabled: False on every rank but the main one.
    """

    def __init__(
        self,
        state_dir: str,
        checkpoint_dir: str,
        interval: int = 10,
        enabled: bool = True,
    ) -> None:
        self.state_dir = state_dir
        self.checkpoint_dir = checkpoint_dir
        self.interval = interval
        self.enabled = enabled
        if enabled:
            os.makedirs(checkpoint_dir, exist_ok=True)

    def write(
        self,
        *,
        epoch: int,
        total_epochs: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        run_id: str | None,
    ) -> None:
        """Write the resume state, and the inference checkpoint when it is due.

        Args:
            epoch: Zero-based index of the epoch just finished.
            total_epochs: The run's horizon, so the last epoch always writes.
            model: The model as it runs, wrappers and all.
            optimizer: The optimizer.
            scheduler: The LR schedule.
            run_id: Logger run to reattach to on resume.
        """
        if not self.enabled:
            return
        save_training_state(
            self.state_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs_completed=epoch + 1,
            wandb_run_id=run_id,
        )
        if (epoch + 1) % self.interval == 0 or (epoch + 1) == total_epochs:
            path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save(unwrap_model(model).state_dict(), path)
            print_main(f"Model saved cleanly to: {path}")


class ResumeState:
    """What :func:`maybe_resume` restored, and where the run starts.

    Attributes:
        start_epoch: Epoch to begin at; 0 for a fresh run.
        run_id: Logger run to reattach to, or ``None``.
    """

    def __init__(self, start_epoch: int = 0, run_id: str | None = None) -> None:
        self.start_epoch = start_epoch
        self.run_id = run_id


def build_dataloader(
    dataset: Any,
    config: TrainingConfig,
    ctx: DistributedContext,
    generator: torch.Generator | None,
) -> tuple[DataLoader[Any], DistributedSampler[Any] | None]:
    """Build the loader, sharding it across ranks only when there are ranks.

    A single-process run keeps the plain shuffle, so its data order is unchanged
    by the existence of this function; ``DistributedSampler`` is introduced only
    where it is needed. Its default padding repeats a few samples to give every
    rank the same number of batches, which is what keeps the ranks in lockstep at
    the last step of the epoch.

    Args:
        dataset: The dataset or split subset.
        config: The training section.
        ctx: Where this process sits in the job.
        generator: Seeded CPU generator for the single-process shuffle.

    Returns:
        ``(dataloader, sampler)``; the sampler is ``None`` off DDP, and the caller
        needs it to call ``set_epoch``.
    """
    sampler: DistributedSampler[Any] | None = None
    if ctx.is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=ctx.world_size,
            rank=ctx.rank,
            shuffle=True,
            seed=config.seed if config.seed is not None else 0,
        )
        print_main(
            f"DDP over {ctx.world_size} ranks: batch_size={config.batch_size} per GPU, "
            f"effective batch {config.batch_size * ctx.world_size}. The learning rate is "
            "NOT rescaled - set training.learning_rate yourself when comparing to a "
            "single-GPU run."
        )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        generator=generator if sampler is None else None,
    )
    return loader, sampler


def build_optimizer_and_scheduler(
    model: torch.nn.Module, config: TrainingConfig
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Build both on the bare module, before any wrapper.

    Neither DDP nor ``torch.compile`` replaces the parameter tensors, so an
    optimizer built here stays loadable by a run that wraps them differently --
    one GPU against eight, compiled against not.

    Args:
        model: The unwrapped model.
        config: The training section, including its scheduler group.

    Returns:
        ``(optimizer, scheduler)``.

    Raises:
        ValueError: If the scheduler type is not one this loop can build. It used
            to be ignored: ``CosineAnnealingLR`` was hardcoded here, so any other
            value trained on cosine anyway.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    # SchedulerConfig validates the name; this asserts the two stay in step, so
    # adding a member there without a branch here fails loudly.
    if config.scheduler.type != "CosineAnnealingLR":
        raise ValueError(f"No builder for training.scheduler.type={config.scheduler.type!r}.")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.scheduler.eta_min
    )
    return optimizer, scheduler


def maybe_resume(
    *,
    state_dir: str,
    device: torch.device,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epochs: int,
    enabled: bool,
) -> ResumeState:
    """Restore weights, optimizer moments and LR position, if a state exists.

    Every rank reads the same file and restores the same state, so no broadcast
    is needed; a rank that somehow missed it fails loudly on the first gradient
    comparison rather than training against stale weights.

    Args:
        state_dir: Directory holding ``last.pt``.
        device: Device to map the stored tensors onto.
        model: Model to load into.
        optimizer: Optimizer to load into.
        scheduler: Schedule to load into.
        epochs: The run's horizon, compared against the stored ``T_max``.
        enabled: ``training.auto_resume``.

    Returns:
        Where to start and which logger run to reattach to.
    """
    if not enabled:
        return ResumeState()

    state = load_training_state(state_dir, device)
    if state is None:
        print_main(f"Auto-resume on, no state at {state_path(state_dir)}: starting fresh.")
        return ResumeState()

    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    # T_max travels inside the scheduler state, so extending training.epochs
    # across a resume would otherwise keep annealing on the old horizon and send
    # the LR back up the far side of the cosine.
    if scheduler.T_max != epochs:
        print_main(
            f"training.epochs changed {scheduler.T_max} -> {epochs} since the "
            "resume state was written; re-anneal on the new horizon."
        )
        scheduler.T_max = epochs

    start_epoch = int(state["epochs_completed"])
    print_main(
        f"Resumed from {state_path(state_dir)} at epoch {start_epoch}/{epochs}. "
        "Weights, optimizer moments and LR schedule restored; the noise and "
        "time draws are not replayed, so the run is not bit-identical to an "
        "uninterrupted one."
    )
    return ResumeState(start_epoch=start_epoch, run_id=state["wandb_run_id"])


def wrap_for_execution(
    model: torch.nn.Module, ctx: DistributedContext, compile_model: bool
) -> torch.nn.Module:
    """Apply DDP and ``torch.compile``, in that order.

    The order is load-bearing: DDPOptimizer needs to see the bucket boundaries,
    which it can only do when the DDP module is the thing being traced.

    Args:
        model: The bare model.
        ctx: Where this process sits in the job.
        compile_model: ``training.compile``.

    Returns:
        The model as it will actually run.
    """
    if ctx.is_distributed:
        model = DistributedDataParallel(
            model, device_ids=[ctx.local_rank] if ctx.device.type == "cuda" else None
        )
    if compile_model:
        print_main("Compiling model via Triton (this may take a minute during the first epoch)...")
        compiled: torch.nn.Module = torch.compile(model)  # type: ignore[assignment]
        return compiled
    print_main("torch.compile disabled (training.compile=false).")
    return model

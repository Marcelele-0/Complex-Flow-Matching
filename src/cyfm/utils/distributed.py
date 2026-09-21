"""Process-group setup and rank helpers for multi-GPU training.

A single-process run goes through these same functions with ``world_size == 1`` and
no process group at all, so ``train.py`` keeps one code path: a laptop run and an
eight-GPU A100 node execute the same program with the same loop.

The rank layout is read from the environment ``torchrun`` exports. Nothing here
knows about Slurm - the batch script's job is to start ``torchrun``, and the moment
it has, this module cannot tell a cluster job from a local ``torchrun`` invocation.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, cast

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# NCCL aborts a rank that waits longer than this for a collective. The 10-minute
# default is shorter than a first epoch that has to scan a cold Lustre directory and
# run the Triton compile, which would kill a healthy job before it reached step one.
COLLECTIVE_TIMEOUT = timedelta(minutes=45)


@dataclass(frozen=True)
class DistributedContext:
    """Where this process sits in the job, and which device it owns."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        """True on the one rank that logs, saves checkpoints and talks to W&B."""
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        """Whether this run has peers. False for a single process, whatever launched it."""
        return self.world_size > 1


def setup_distributed() -> DistributedContext:
    """Join the ``torchrun`` process group, or return a single-process context.

    ``torchrun`` exports ``RANK``/``LOCAL_RANK``/``WORLD_SIZE`` and the rendezvous
    address; their absence is what marks a plain ``python -m cyfm.train`` run.

    Returns:
        The rank layout and this process's device. The process group is initialized
        as a side effect when ``world_size > 1``.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        # Pins every later allocation and collective to this rank's GPU. Without it
        # all ranks default to cuda:0, which oversubscribes one card and deadlocks
        # NCCL.
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if world_size > 1:
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend, timeout=COLLECTIVE_TIMEOUT)

    return DistributedContext(
        rank=rank, local_rank=local_rank, world_size=world_size, device=device
    )


def cleanup_distributed() -> None:
    """Tear the process group down, so a clean exit does not warn about a leak."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """True on rank 0, readable before :func:`setup_distributed` has run."""
    return int(os.environ.get("RANK", "0")) == 0


def print_main(*args: Any, **kwargs: Any) -> None:
    """``print`` on rank 0 only, so eight ranks do not write eight copies of a line."""
    if is_main_process():
        print(*args, **kwargs)


def sum_across_ranks(values: Sequence[float], device: torch.device) -> list[float]:
    """Element-wise sum of ``values`` over every rank; identity when not distributed.

    Used for epoch totals: each rank sees only its shard, so the printed average is
    the average over that shard unless the numerator and the batch count are both
    reduced first.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return list(values)
    # float64 because these are running sums over a whole epoch, where float32 loses
    # low-order digits once the total is a few orders of magnitude above one term.
    totals = torch.tensor(list(values), dtype=torch.float64, device=device)
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return cast(list[float], totals.tolist())


def any_across_ranks(flag: bool, device: torch.device) -> bool:
    """True if ``flag`` is set on any rank; identity when not distributed.

    Slurm's grace signal does not reliably reach every worker through the launcher.
    A rank that stopped on a signal its peers never saw would leave them blocked in
    the next collective until the NCCL timeout, so the decision to stop is agreed on
    rather than taken locally.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return flag
    vote = torch.tensor([int(flag)], dtype=torch.int32, device=device)
    dist.all_reduce(vote, op=dist.ReduceOp.MAX)
    return bool(vote.item())


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Strip ``DistributedDataParallel`` and ``torch.compile`` wrappers, in any order.

    Checkpoints have to hold the bare module's keys: ``evaluate.py``,
    ``evaluate.py`` load them into a model that is neither compiled nor wrapped, and
    a stored ``module.``/``_orig_mod.`` prefix would make that fail.
    """
    inner = model
    while True:
        if isinstance(inner, DistributedDataParallel):
            inner = inner.module
        elif hasattr(inner, "_orig_mod"):
            inner = cast(torch.nn.Module, inner._orig_mod)
        else:
            return inner

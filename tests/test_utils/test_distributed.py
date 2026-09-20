"""Rank plumbing that has to behave identically with and without a process group.

The single-process path is the one every laptop run and every unit test takes, so
these check that the distributed helpers degrade to identities rather than to
errors - a helper that only works under torchrun would make train.py two programs.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from cyfm.utils.distributed import (
    any_across_ranks,
    is_main_process,
    print_main,
    setup_distributed,
    sum_across_ranks,
    unwrap_model,
)

CPU = torch.device("cpu")


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class FakeCompiled(nn.Module):
    """Stands in for torch.compile's OptimizedModule, which exposes ``_orig_mod``.

    Compiling for real in a unit test costs a Triton build and needs a backend; the
    only contract unwrap_model relies on is the attribute.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self._orig_mod = inner


@pytest.fixture
def single_rank_group():
    """A real one-process gloo group, so DDP can actually be constructed."""
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29677",
        rank=0,
        world_size=1,
    )
    yield
    dist.destroy_process_group()


def test_setup_distributed_without_torchrun_is_single_process(monkeypatch) -> None:
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(key, raising=False)

    ctx = setup_distributed()

    assert (ctx.rank, ctx.local_rank, ctx.world_size) == (0, 0, 1)
    assert ctx.is_main
    assert not ctx.is_distributed
    assert not dist.is_initialized()


def test_setup_distributed_reads_the_torchrun_layout(monkeypatch) -> None:
    # world_size stays 1 so no process group is created; the point is that the rank
    # fields come from the environment torchrun exports rather than being assumed.
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")

    ctx = setup_distributed()

    assert (ctx.rank, ctx.local_rank, ctx.world_size) == (0, 0, 1)


def test_is_main_process_follows_rank(monkeypatch) -> None:
    monkeypatch.delenv("RANK", raising=False)
    assert is_main_process()

    monkeypatch.setenv("RANK", "3")
    assert not is_main_process()


def test_print_main_is_silent_off_rank_zero(monkeypatch, capsys) -> None:
    monkeypatch.setenv("RANK", "1")
    print_main("should not appear")
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("RANK", "0")
    print_main("should appear")
    assert "should appear" in capsys.readouterr().out


def test_reductions_are_identities_without_a_group() -> None:
    assert not dist.is_initialized()
    assert sum_across_ranks([1.5, 2.0, 8.0], CPU) == [1.5, 2.0, 8.0]
    assert any_across_ranks(True, CPU) is True
    assert any_across_ranks(False, CPU) is False


def test_reductions_run_over_a_real_group(single_rank_group) -> None:
    # One rank makes the sum an identity numerically, but it exercises the collective
    # rather than the early return, which is the branch torchrun actually takes.
    assert sum_across_ranks([1.5, 2.5], CPU) == [1.5, 2.5]
    assert any_across_ranks(False, CPU) is False
    assert any_across_ranks(True, CPU) is True


def test_unwrap_model_returns_a_bare_module_unchanged() -> None:
    model = TinyNet()
    assert unwrap_model(model) is model


def test_unwrap_model_strips_ddp(single_rank_group) -> None:
    model = TinyNet()
    assert unwrap_model(DistributedDataParallel(model)) is model


def test_unwrap_model_strips_compile_over_ddp(single_rank_group) -> None:
    # The order train.py uses: DDP first so DDPOptimizer sees the buckets, then
    # compile, which leaves both wrappers between the checkpoint and the weights.
    model = TinyNet()
    assert unwrap_model(FakeCompiled(DistributedDataParallel(model))) is model


def test_unwrap_model_strips_compile_alone() -> None:
    model = TinyNet()
    assert unwrap_model(FakeCompiled(model)) is model


def test_wrapped_state_dict_keys_match_the_bare_model(single_rank_group) -> None:
    # This is the property evaluate.py depends on: a checkpoint from
    # an eight-GPU compiled run has to load into a bare, uncompiled model.
    model = TinyNet()
    wrapped = FakeCompiled(DistributedDataParallel(model))

    assert set(unwrap_model(wrapped).state_dict()) == set(model.state_dict())
    assert not any(
        key.startswith(("module.", "_orig_mod.")) for key in unwrap_model(wrapped).state_dict()
    )


def test_setup_distributed_leaves_no_env_residue(monkeypatch) -> None:
    # setup_distributed reads the environment; it must not write to it, or a second
    # entry point in the same process would inherit a rank it never asked for.
    monkeypatch.delenv("RANK", raising=False)
    before = dict(os.environ)

    setup_distributed()

    assert dict(os.environ) == before

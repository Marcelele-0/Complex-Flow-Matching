"""Find the largest training batch one GPU holds at a protocol's field size.

Issue #76: the synthetic fields trained at batch 64 on 64x64, and 320x320 is 25x the
pixels, so that batch does not carry over to fastMRI. The batch size is part of the
protocol -- every arm of a table has to share it -- so it is measured, not guessed.

Each trial runs full training steps built the way ``cyfm.train`` builds them, from the
same experiment config: the prior draw, the coupling, the bridge, the forward pass,
the loss, backward and an AdamW step. The optimizer's moments and the coupling's cost
matrix are therefore both inside the measurement. The data states are synthetic, so
no store is needed and the probe runs before any data reaches the cluster.

``--memory-gib`` caps PyTorch's allocator, which lets an H100 answer for a smaller
card. The cap does not cover the CUDA context a real card also has to hold, so give it
a little less than the card's size: 15 for a 16 GB card.

Usage::

    uv run python scripts/paper/probe_batch_size.py --side 320 \
        +experiment=table5_fastmri manifold=cylindrical training.coupling=ot
    uv run python scripts/paper/probe_batch_size.py --side 320 --memory-gib 15 \
        +experiment=table5_fastmri manifold=euclidean training.coupling=ot
"""

from __future__ import annotations

import argparse
import gc
import pathlib
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from hydra import compose, initialize_config_dir

from cyfm.config.adapters import manifold_from_config
from cyfm.flow.couplings import build_coupling
from cyfm.utils.inference import build_model

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONF = ROOT / "conf"
GIB = 2**30


@dataclass
class Trial:
    """One batch size's outcome."""

    batch: int
    fits: bool
    peak_allocated_gib: float
    peak_reserved_gib: float
    seconds_per_step: float


def largest_fitting(fits: Callable[[int], bool], max_batch: int) -> int:
    """The largest batch in ``[1, max_batch]`` for which ``fits`` holds.

    Doubles until the first failure, then bisects between the last success and it, so
    a card that holds ~100 costs about fourteen trials rather than a hundred. Assumes
    ``fits`` is monotone, which memory is.

    Args:
        fits: Runs a trial at a batch size and reports whether it completed.
        max_batch: Ceiling on the search.

    Returns:
        The largest fitting batch, or ``0`` when not even a batch of one fits.
    """
    if not fits(1):
        return 0
    good = 1
    bad: int | None = None
    while bad is None:
        trial = min(good * 2, max_batch)
        if trial == good:
            return good
        if fits(trial):
            good = trial
        else:
            bad = trial
    while bad - good > 1:
        middle = (good + bad) // 2
        if fits(middle):
            good = middle
        else:
            bad = middle
    return good


def largest_power_of_two(n: int) -> int:
    """The largest power of two not above ``n``, or ``0``."""
    return 1 << (n.bit_length() - 1) if n > 0 else 0


def _out_of_memory(error: BaseException) -> bool:
    # The allocator raises OutOfMemoryError, but cuDNN and cuBLAS workspaces fail as
    # plain RuntimeErrors carrying the same message.
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def build_trial(
    overrides: Sequence[str], side: int, device: torch.device, steps: int
) -> Callable[[int], Trial]:
    """Build the model, geometry, coupling and optimizer once, and return a trial runner.

    Args:
        overrides: Hydra overrides selecting the experiment, e.g. ``+experiment=...``.
        side: Field height and width.
        device: Where to run.
        steps: Training steps per trial; the last one is timed.

    Returns:
        A function running ``steps`` training steps at a batch size.
    """
    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        cfg = compose(config_name="config", overrides=list(overrides))

    manifold = manifold_from_config(cfg).to(device)
    model = build_model(
        cfg,
        device,
        in_channels=manifold.state_channels,
        out_channels=manifold.velocity_channels,
    )
    model.train()
    training = cfg.get("training", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.get("learning_rate", 2e-4),
        weight_decay=training.get("weight_decay", 1e-4),
    )
    coupling_name = str(training.get("coupling", "independent"))
    coupling = build_coupling(coupling_name)
    reorders = coupling_name not in ("independent", "none")
    grad_clip = training.get("grad_clip", 1.0)
    clip = float("inf") if grad_clip is None else float(grad_clip)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"manifold={manifold.name} coupling={coupling_name} model={type(model).__name__} "
        f"parameters={parameters} side={side}",
        flush=True,
    )

    def step(batch: int) -> None:
        # Synthetic states stand in for the store: the same shape and dtype as the
        # fields training reads, which is all that memory and speed depend on.
        x_1 = manifold.sample_noise(batch, side, side, device)
        x_0 = manifold.sample_noise(batch, side, side, device)
        t = torch.rand(batch, device=device)
        if reorders:
            x_1 = coupling(x_0, x_1, manifold)
        x_t, target_v = manifold.bridge(x_0, x_1, t.view(batch, 1, 1, 1))
        optimizer.zero_grad()
        pred_v = model(x_t, t)
        loss, _ = manifold.loss(pred_v, target_v, target_x1=x_1, t=t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    def trial(batch: int) -> Trial:
        cuda = device.type == "cuda"
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        fits = True
        seconds = float("nan")
        try:
            for index in range(steps):
                started = time.perf_counter()
                step(batch)
                if cuda:
                    torch.cuda.synchronize(device)
                if index == steps - 1:
                    seconds = time.perf_counter() - started
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            if not _out_of_memory(error):
                raise
            fits = False
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        allocated = torch.cuda.max_memory_allocated(device) / GIB if cuda else float("nan")
        reserved = torch.cuda.max_memory_reserved(device) / GIB if cuda else float("nan")
        if cuda:
            torch.cuda.empty_cache()
        return Trial(batch, fits, allocated, reserved, seconds)

    return trial


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--side", type=int, default=320, help="Field height and width.")
    parser.add_argument("--max-batch", type=int, default=256, help="Ceiling on the search.")
    parser.add_argument(
        "--memory-gib",
        type=float,
        default=0.0,
        help="Cap the allocator at this many GiB to stand in for a smaller card; 0 = no cap.",
    )
    parser.add_argument("--steps", type=int, default=2, help="Steps per trial; the last is timed.")
    parser.add_argument("--device", default="cuda", help="cuda, or cpu to check wiring only.")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. +experiment=...")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            print(
                "no CUDA device: a batch-size probe on CPU measures nothing. Run it on a "
                "GPU node, e.g. through scripts/wcss/gate_fastmri.sbatch.",
                file=sys.stderr,
            )
            return 2
        total = torch.cuda.get_device_properties(device).total_memory
        if args.memory_gib > 0:
            # torch wants an index here, not the string "cuda": passing the bare
            # string raises ValueError and the capped probe never runs.
            torch.cuda.set_per_process_memory_fraction(
                min(1.0, args.memory_gib * GIB / total), torch.cuda.current_device()
            )
        capacity = args.memory_gib if args.memory_gib > 0 else total / GIB
        print(f"device={torch.cuda.get_device_name(device)} capacity_gib={capacity:.1f}")
    else:
        capacity = float("nan")
        print("device=cpu: wiring check only, memory is not measured")

    trial = build_trial(args.overrides, args.side, device, max(1, args.steps))

    def fits(batch: int) -> bool:
        record = trial(batch)
        print(
            f"batch={batch:<5} fits={'yes' if record.fits else 'no':<4} "
            f"peak_allocated_gib={record.peak_allocated_gib:6.2f} "
            f"peak_reserved_gib={record.peak_reserved_gib:6.2f} "
            f"seconds_per_step={record.seconds_per_step:.3f}",
            flush=True,
        )
        return record.fits

    best = largest_fitting(fits, args.max_batch)
    print(
        f"\nside={args.side} max_batch={best} max_power_of_two={largest_power_of_two(best)} "
        f"capacity_gib={capacity:.1f}"
    )
    if best == 0:
        return 1
    if best == args.max_batch:
        print(
            f"max_batch hit the --max-batch ceiling ({args.max_batch}); the card was not the limit"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

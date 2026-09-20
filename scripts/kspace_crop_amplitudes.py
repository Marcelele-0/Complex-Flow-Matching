"""What a k-space centre crop does to the knee amplitude law, before any of it reaches a table.

The reduced-resolution knee run exists to test one hypothesis: that the rise in sliced $W_2$
from $k=1$ to $k=2$ survives joint OT on knee MRI because minibatch OT's transport-cost
reduction collapses with field dimension. Running at 64x64, where OT still buys something,
is supposed to separate that from the alternative.

**It only separates them if the crop leaves the amplitude law intact.** At 320x320, 17.59%
of knee coefficients are exactly zero, with phase a placeholder out of ``atan2(0, 0)``, and
the competing explanation for the rise is precisely that bimodal law with its spike at the
origin. Truncating the spectrum convolves the image with a sinc kernel, so hard zeros smear.
If the crop removes them, a rise that disappears at 64x64 has two explanations and the
experiment settles nothing.

So this script measures the thing that would confound the result, and it measures it before
the training runs rather than after. Network-free, like ``scripts/coupling_dimension.py``:
every number here is a property of the data and the crop.

Amplitudes are reported in the training domain --- each field divided by its own peak
modulus, through the manifold's own pipeline --- because that is the domain the tables score
in. Exact zeros are scale-invariant, so their fraction does not depend on that choice.

Usage::

    uv run python scripts/kspace_crop_amplitudes.py
    uv run python scripts/kspace_crop_amplitudes.py --sizes 320 128 64 --num-fields 256
    uv run python scripts/kspace_crop_amplitudes.py --store train.h5 --role fit
"""

from __future__ import annotations

import argparse
from typing import Any

import torch

from cyfm.data.stores.knee import KneeStoreDataset
from cyfm.data.transforms import slice_transform
from cyfm.manifolds import build_manifold
from cyfm.metrics import circular_linear_correlation

_LOW_THRESHOLDS = (0.0, 1e-6, 1e-4, 1e-3, 1e-2)
_QUANTILES = (0.01, 0.05, 0.25, 0.50)


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default="data/knee_pd", help="Store directory.")
    parser.add_argument("--store", type=str, default="val.h5", help="Store file name.")
    parser.add_argument(
        "--role",
        type=str,
        default="all",
        help="Volume role. Default 'all': this is a property of the data, not of a split.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[320, 128, 64],
        help="Matrices to report. The first should be the store's own, as the baseline.",
    )
    parser.add_argument(
        "--num-fields", type=int, default=256, help="Slices drawn, evenly spaced through the store."
    )
    return parser.parse_args()


def cylinder() -> Any:
    """The cylindrical manifold, built the way the entry points build it."""
    return build_manifold({"name": "cylindrical", "spatial_correlation": None})


def amplitudes_and_phases(
    args: argparse.Namespace, manifold: Any, size: int | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pooled amplitude and phase of the cohort at one matrix size.

    Args:
        args: Parsed arguments.
        manifold: Geometry supplying the representation pipeline.
        size: k-space crop to apply, or ``None`` for the store's own matrix.

    Returns:
        Flat amplitude and phase tensors, in the training domain.
    """
    dataset = KneeStoreDataset(
        data_dir=args.data_dir,
        store=args.store,
        role=args.role,
        kspace_crop=size,
        transform=slice_transform(manifold, crop_base=16),
    )
    step = max(1, len(dataset) // args.num_fields)
    states = torch.stack([dataset[i] for i in range(0, len(dataset), step)][: args.num_fields])
    amplitude = states[:, 0].reshape(-1)
    phase = torch.atan2(states[:, 2], states[:, 1]).reshape(-1)
    return amplitude, phase


def report(size: int, amplitude: torch.Tensor, phase: torch.Tensor) -> float:
    """Print one size's row and return its exact-zero fraction."""
    total = amplitude.numel()
    zero_fraction = float((amplitude == 0.0).sum()) / total
    below = [float((amplitude <= t).sum()) / total for t in _LOW_THRESHOLDS[1:]]
    quantiles = torch.quantile(amplitude.float(), torch.tensor(_QUANTILES))
    dependence = circular_linear_correlation(amplitude, phase)
    # The zero background carries phase 0.0 from atan2(0, 0) -- a placeholder, not a
    # measurement -- and 17% of coefficients piled at (0, 0) manufacture a dependence that
    # is not in the tissue. Over the val store the raw figure sits near 0.30 while the
    # masked one is under 0.05 and moves with the slice subset, so read the masked column
    # as "weak" rather than as a value. Never quote the unmasked one as a property of knee
    # tissue.
    kept = amplitude > 0.0
    masked = circular_linear_correlation(amplitude[kept], phase[kept])

    print(f"\n{size}x{size}  ({total:,} coefficients)")
    print(f"  exactly zero        {zero_fraction:>8.4%}")
    for threshold, share in zip(_LOW_THRESHOLDS[1:], below, strict=True):
        print(f"  <= {threshold:<8g}        {share:>8.4%}")
    for q, value in zip(_QUANTILES, quantiles.tolist(), strict=True):
        print(f"  q{q:<4.2f} amplitude     {value:>8.5f}")
    print(f"  mean amplitude      {float(amplitude.mean()):>8.5f}")
    print(f"  circular-linear r   {dependence:>8.5f}  (all pixels)")
    print(f"  circular-linear r   {masked:>8.5f}  (exact zeros dropped)")
    return zero_fraction


def main() -> None:
    """Measure the amplitude law at each requested matrix size."""
    args = parse_args()
    manifold = cylinder()

    print(f"{args.store} role={args.role}, up to {args.num_fields} slices per size")
    fractions: dict[int, float] = {}
    for index, size in enumerate(args.sizes):
        amplitude, phase = amplitudes_and_phases(args, manifold, None if index == 0 else size)
        fractions[size] = report(size, amplitude, phase)

    baseline = fractions[args.sizes[0]]
    print("\n" + "=" * 60)
    for size, fraction in fractions.items():
        print(f"  {size:>4}x{size:<4} exact zeros {fraction:>8.4%}")
    survivors = [s for s, f in fractions.items() if s != args.sizes[0] and f > 0.5 * baseline]
    if survivors:
        print(f"\nThe zero spike survives at {survivors}: dimension and the amplitude law")
        print("stay separable there, and a reduced-size run answers the step-3 hypothesis.")
    else:
        print(f"\nThe zero spike does NOT survive the crop (baseline {baseline:.4%}).")
        print("A rise that disappears at a reduced size then has two explanations, and this")
        print("experiment alone cannot attribute it to dimension. A zero-preserving control")
        print("arm at the same matrix is needed before the result goes in the paper.")


if __name__ == "__main__":
    main()

"""Table 4 / Section 5.4 -- what factorising the coupling across patches destroys.

Couples 16x16 patches of 64x64 fields independently and measures what survives:
the pooled sliced W2 to the data, the lag-1 correlation inside patches, and the
lag-1 correlation across patch seams. The paper's claim is that the first two are
untouched while the third collapses.

Runs in about a minute on a CPU and needs no data.

    uv run python -m reproducibility.table4_patch_seams
    uv run python -m reproducibility.table4_patch_seams --seeds 8   # the spread

**Read the seam line carefully.** The paper prints ``0.044`` for it. That is one
draw: across eight seeds the statistic ranges over roughly ``[-0.008, +0.055]``,
and the seed the shipped command uses gives ``-0.008``. The claim reproduces
under every seed; the digit does not reproduce under the shipped one. This script
reports both, which is the point of it.
"""

from __future__ import annotations

import argparse

import torch

from cyfm.data.toy import CylinderToyFieldDataset
from cyfm.flow.couplings import OptimalTransportCoupling
from cyfm.flow.transport import sliced_wasserstein2
from cyfm.manifolds import build_manifold
from reproducibility._harness import Check, Outcome, compare, exit_with, report
from reproducibility.expected import SEAM_LAG1

SIDE, PATCH, BATCH = 64, 16, 64
CORRELATION_LENGTH = 6.0


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Seeds to measure. 1 reproduces the shipped command; 8 shows the spread.",
    )
    return parser.parse_args()


def lag_one_at(fields: torch.Tensor, columns: list[int]) -> float:
    """Lag-one correlation of the amplitude field across the given columns.

    Args:
        fields: Complex fields ``[B, 1, H, W]``.
        columns: Column indices whose correlation with their left neighbour is
            measured. Patch boundaries for the seam statistic, patch interiors
            for the control.

    Returns:
        The Pearson correlation over every measured pair.
    """
    amplitude = fields.abs()
    left = torch.cat([amplitude[..., c - 1 : c] for c in columns], dim=-1).reshape(-1)
    right = torch.cat([amplitude[..., c : c + 1] for c in columns], dim=-1).reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    denominator = (left.norm() * right.norm()).clamp_min(1e-12)
    return float((left * right).sum() / denominator)


def measure(seed: int) -> dict[str, float]:
    """Couple patches independently and measure what it cost.

    Args:
        seed: Seeds the field draw and the prior.

    Returns:
        ``seam_lag1``, ``inside_lag1`` and ``sliced_w2`` after patch-level
        coupling.
    """
    manifold = build_manifold({"name": "cylindrical", "spatial_correlation": None})
    coupling = OptimalTransportCoupling()
    device = torch.device("cpu")

    dataset = CylinderToyFieldDataset(
        coupling=0.5,
        size=BATCH,
        crop_size=(SIDE, SIDE),
        correlation_length=CORRELATION_LENGTH,
        seed=seed,
    )
    raw = torch.stack([dataset[i] for i in range(BATCH)])
    data = manifold.from_complex(raw).to(device)
    prior = manifold.sample_noise(
        BATCH, SIDE, SIDE, device, generator=torch.Generator(device=device).manual_seed(seed)
    )

    patched = data.clone()
    for row in range(SIDE // PATCH):
        for column in range(SIDE // PATCH):
            window = (
                slice(None),
                slice(None),
                slice(row * PATCH, (row + 1) * PATCH),
                slice(column * PATCH, (column + 1) * PATCH),
            )
            patched[window] = coupling(
                prior[window].contiguous(), data[window].contiguous(), manifold
            )
    patchwork = manifold.to_complex(patched).cpu()

    seams = [PATCH * k for k in range(1, SIDE // PATCH)]
    inside = [PATCH * k + PATCH // 2 for k in range(SIDE // PATCH)]
    cloud = torch.stack([patchwork.reshape(-1).real, patchwork.reshape(-1).imag], 1)
    reference = torch.stack([raw.reshape(-1).real, raw.reshape(-1).imag], 1)
    return {
        "seam_lag1": lag_one_at(patchwork, seams),
        "inside_lag1": lag_one_at(patchwork, inside),
        "sliced_w2": float(
            sliced_wasserstein2(cloud, reference, 256, generator=torch.Generator().manual_seed(1))
        ),
    }


def main() -> int:
    """Measure at the shipped seed, and report the spread when asked for more."""
    args = parse_args()
    measurements = [measure(seed) for seed in range(args.seeds)]
    shipped = measurements[0]

    checks: list[Check] = [
        compare("lag-1 inside patches", shipped["inside_lag1"], SEAM_LAG1["inside_patches"]),
        compare("pooled sliced W2 to data", shipped["sliced_w2"], SEAM_LAG1["sliced_w2"]),
        compare(
            "lag-1 across seams (published digit)",
            shipped["seam_lag1"],
            SEAM_LAG1["published"],
        ),
    ]

    # The claim, as opposed to the digit: seam correlation collapses to about
    # zero. This is what holds under every seed, and it is the assertion worth
    # failing on.
    collapse = SEAM_LAG1["collapse_below"]
    worst = max(abs(m["seam_lag1"]) for m in measurements)
    checks.append(
        Check(
            label="seam correlation collapses (the claim)",
            outcome=Outcome.PASS if worst <= collapse.value else Outcome.FAIL,
            measured=worst,
            expectation=collapse,
            detail=(
                f"|seam lag-1| at worst {worst:.4f} over {args.seeds} seed(s), "
                f"against 0.979 before coupling; bound {collapse.value:.2f}"
            ),
        )
    )

    notes = []
    if args.seeds > 1:
        values = torch.tensor([m["seam_lag1"] for m in measurements])
        notes.append(
            f"Seam lag-1 over {args.seeds} seeds: mean {values.mean():+.4f}, "
            f"sd {values.std():.4f}, range [{values.min():+.4f}, {values.max():+.4f}].\n"
            "The paper's 0.044 is one draw from this, and so is the shipped seed's value."
        )
    else:
        notes.append(
            "Run with --seeds 8 to see the spread this statistic has. The paper prints one\n"
            "draw of it in a sentence whose other two numbers are stable to three decimals."
        )

    return report("TABLE 4 / SECTION 5.4 -- patch seams", checks, notes=notes)


if __name__ == "__main__":
    exit_with(main())

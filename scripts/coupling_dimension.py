"""How much minibatch optimal transport buys as fields grow, and what patching costs.

Network-free, like the gate scripts: every number here is a property of the
couplings and the data, not of anything trained. Five measurements.

1. **Cost reduction against field size.** The joint assignment over whole
   fields, against the pairing the dataloader drew. Concentration of measure
   predicts the gain falls as the field grows, because pairwise costs between
   high-dimensional samples all approach one value.
2. **Against batch size.** Whether a larger batch buys the gain back.
3. **Against data smoothness.** Whether spatially correlated *data* lowers the
   effective dimension enough to help. With a white prior it should not: the
   displacement inherits the prior's whiteness.
4. **Against prior smoothness.** The same question from the prior's side, with
   the mean angular displacement the cylinder has to learn.
5. **Patch-level coupling.** An independent assignment per patch position gains
   dimension by giving each position its own permutation, which assembles every
   endpoint from different samples. Spatial correlation across seams is the
   observable that exposes it; a pooled distributional metric cannot.

Usage::

    uv run python scripts/coupling_dimension.py
    uv run python scripts/coupling_dimension.py --seed 1
"""

from __future__ import annotations

import argparse
import math

import torch
from omegaconf import OmegaConf

from cyfm.core.manifold import BaseManifold
from cyfm.data.toy_dataset import CylinderToyFieldDataset, CylinderToyIIDDataset
from cyfm.flow.couplings import OptimalTransportCoupling
from cyfm.flow.transport import sliced_wasserstein2
from cyfm.manifolds import build_manifold


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="Seed for priors and data.")
    parser.add_argument("--device", type=str, default=None, help="Override the device.")
    return parser.parse_args()


def cylinder(spatial_correlation: float | None = None) -> BaseManifold:
    """The cylindrical manifold, built the way the entry points build it."""
    return build_manifold(
        OmegaConf.create(
            {
                "manifold": {"name": "cylindrical", "spatial_correlation": spatial_correlation},
                "training": {"loss": {}},
            }
        )
    )


def fields(dataset: CylinderToyIIDDataset | CylinderToyFieldDataset, count: int) -> torch.Tensor:
    """The first ``count`` fields of a dataset, complex ``[B, 1, H, W]``."""
    return torch.stack([dataset[index] for index in range(count)])


def pairing_cost(manifold: BaseManifold, prior: torch.Tensor, data: torch.Tensor) -> float:
    """Mean weighted squared geodesic displacement of an aligned pairing."""
    weights = manifold.tangent_weights.to(prior.device).reshape(1, -1, 1, 1)
    displacement = manifold.log_map(prior, data)
    return float((displacement.square() * weights).flatten(1).mean(1).mean())


def draw_prior(
    manifold: BaseManifold, batch: int, side: int, device: torch.device, seed: int
) -> torch.Tensor:
    """A reproducible prior batch."""
    generator = torch.Generator(device=device).manual_seed(seed)
    return manifold.sample_noise(batch, side, side, device, generator=generator)


def lag_one_at(field: torch.Tensor, columns: list[int]) -> float:
    """Correlation of amplitudes on either side of the given column boundaries."""
    amplitude = field.abs()[:, 0]
    left = torch.cat([amplitude[:, :, c - 1].reshape(-1) for c in columns])
    right = torch.cat([amplitude[:, :, c].reshape(-1) for c in columns])
    left, right = left - left.mean(), right - right.mean()
    return float((left * right).sum() / (left.norm() * right.norm()))


def main() -> None:
    """Run all five measurements."""
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    seed = args.seed
    coupling = OptimalTransportCoupling()
    manifold = cylinder()

    print("=" * 96)
    print("1. OT COST REDUCTION vs FIELD SIZE   cylinder_toy_iid rho 0.5, batch 64, white prior")
    print("=" * 96)
    print(f"{'field':>8}{'dim':>7}{'cost drop':>12}{'reordered':>12}{'cost spread':>14}")
    # 320 is the knee acquisition's own side, and the only size in this table where
    # a paper table is computed on real data rather than synthetic fields.
    for side in (1, 2, 4, 8, 16, 32, 64, 128, 320):
        data = manifold.from_complex(
            fields(
                CylinderToyIIDDataset(coupling=0.5, size=64, crop_size=(side, side), seed=seed), 64
            )
        ).to(device)
        prior = draw_prior(manifold, 64, side, device, seed)
        cost = coupling.cost_matrix(prior, data, manifold)
        paired = coupling(prior, data, manifold)
        reordered = float((~torch.isclose(paired, data).flatten(1).all(1)).double().mean())
        drop = 1.0 - pairing_cost(manifold, prior, paired) / pairing_cost(manifold, prior, data)
        spread = float(cost.std() / cost.mean())
        print(
            f"{f'{side}x{side}':>8}{side * side:>7}{drop:>11.1%}{reordered:>11.1%}{spread:>14.4f}"
        )

    print("\n" + "=" * 96)
    print("2. OT COST REDUCTION vs BATCH SIZE   16x16, cylinder_toy_iid rho 0.5, white prior")
    print("=" * 96)
    for batch in (8, 16, 32, 64, 128, 256):
        data = manifold.from_complex(
            fields(
                CylinderToyIIDDataset(coupling=0.5, size=batch, crop_size=(16, 16), seed=seed),
                batch,
            )
        ).to(device)
        prior = draw_prior(manifold, batch, 16, device, seed)
        drop = 1.0 - pairing_cost(manifold, prior, coupling(prior, data, manifold)) / pairing_cost(
            manifold, prior, data
        )
        print(f"  batch {batch:>4}: cost drop {drop:.1%}")

    print("\n" + "=" * 96)
    print("3. SMOOTH DATA, WHITE PRIOR   64x64, batch 64")
    print("=" * 96)
    for length in (0.0, 4.0, 16.0, 32.0, 63.0):
        data = manifold.from_complex(
            fields(
                CylinderToyFieldDataset(
                    coupling=0.5, size=64, crop_size=(64, 64), correlation_length=length, seed=seed
                ),
                64,
            )
        ).to(device)
        prior = draw_prior(manifold, 64, 64, device, seed)
        drop = 1.0 - pairing_cost(manifold, prior, coupling(prior, data, manifold)) / pairing_cost(
            manifold, prior, data
        )
        print(f"  data correlation length {length:>4.0f}: cost drop {drop:.1%}")

    print("\n" + "=" * 96)
    print("4. SMOOTH PRIOR   cylinder_toy_field correlation 4, 64x64")
    print("=" * 96)
    print(
        f"{'batch':>6}{'prior corr':>12}{'amp std':>9}{'cost drop':>11}"
        f"{'|u_phi| indep':>15}{'|u_phi| OT':>12}"
    )
    for batch in (64, 256):
        raw = fields(
            CylinderToyFieldDataset(
                coupling=0.5, size=batch, crop_size=(64, 64), correlation_length=4.0, seed=seed
            ),
            batch,
        )
        for prior_length in (None, 2.0, 4.0, 8.0, 16.0):
            smooth = cylinder(prior_length)
            data = smooth.from_complex(raw).to(device)
            prior = draw_prior(smooth, batch, 64, device, seed + 7)
            paired = coupling(prior, data, smooth)
            independent, coupled = smooth.log_map(prior, data), smooth.log_map(prior, paired)
            drop = 1.0 - pairing_cost(smooth, prior, paired) / pairing_cost(smooth, prior, data)
            print(
                f"{batch:>6}{str(prior_length):>12}{float(prior[:, 0].std()):>9.3f}{drop:>10.1%}"
                f"{float(independent[:, 1].abs().mean()):>15.3f}"
                f"{float(coupled[:, 1].abs().mean()):>12.3f}"
            )
    print(
        f"  reference: uniform modulus std {1 / math.sqrt(12):.3f}; "
        f"|u_phi| uniform = {math.pi / 2:.3f}"
    )

    print("\n" + "=" * 96)
    print("5. PATCH-LEVEL COUPLING   cylinder_toy_field correlation 6, 64x64, 16x16 patches")
    print("=" * 96)
    side, patch, batch = 64, 16, 64
    raw = fields(
        CylinderToyFieldDataset(
            coupling=0.5, size=batch, crop_size=(side, side), correlation_length=6.0, seed=seed
        ),
        batch,
    )
    data = manifold.from_complex(raw).to(device)
    prior = draw_prior(manifold, batch, side, device, seed)
    whole = manifold.to_complex(coupling(prior, data, manifold)).cpu()
    patched = data.clone()
    for row in range(side // patch):
        for column in range(side // patch):
            window = (
                slice(None),
                slice(None),
                slice(row * patch, (row + 1) * patch),
                slice(column * patch, (column + 1) * patch),
            )
            patched[window] = coupling(
                prior[window].contiguous(), data[window].contiguous(), manifold
            )
    patchwork = manifold.to_complex(patched).cpu()

    seams = [patch * k for k in range(1, side // patch)]
    inside = [patch * k + patch // 2 for k in range(side // patch)]
    reference = torch.stack([raw.reshape(-1).real, raw.reshape(-1).imag], 1)
    print(
        f"{'endpoints':<26}{'lag1 across seams':>19}"
        f"{'lag1 inside patches':>21}{'sliced W2 to data':>19}"
    )
    for label, field in (
        ("data (untouched)", raw),
        ("image-level OT", whole),
        ("patch-level OT", patchwork),
    ):
        cloud = torch.stack([field.reshape(-1).real, field.reshape(-1).imag], 1)
        distance = float(
            sliced_wasserstein2(cloud, reference, 256, generator=torch.Generator().manual_seed(1))
        )
        print(
            f"{label:<26}{lag_one_at(field, seams):>19.3f}"
            f"{lag_one_at(field, inside):>21.3f}{distance:>19.4f}"
        )


if __name__ == "__main__":
    main()

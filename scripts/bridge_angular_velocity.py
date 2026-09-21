"""Angular velocity of the two bridges, measured on the bridges and not on a network.

Command-line front end for :class:`cyfm.experiments.BridgeGeometryExperiment`,
which holds the measurement and the argument for it. This file stays because
``conf/experiment/table1_bridges.yaml`` names it by path and
``tests/test_experiments.py`` asserts that path exists.

Usage::

    uv run python scripts/bridge_angular_velocity.py
    uv run python scripts/bridge_angular_velocity.py --pairs 2000000 --coupling 0.5
    uv run python scripts/bridge_angular_velocity.py --json outputs/table1.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

from cyfm.experiments.bridge_geometry import BridgeGeometryExperiment, render


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=1_000_000, help="Endpoint pairs to draw.")
    parser.add_argument(
        "--coupling", type=float, default=0.5, help="Amplitude-phase dependence of the target."
    )
    parser.add_argument(
        "--ot-batch", type=int, default=256, help="Batch size for the minibatch OT arm."
    )
    parser.add_argument(
        "--target-store",
        type=pathlib.Path,
        default=None,
        help="Draw the target from this HDF5 store instead of the synthetic toy.",
    )
    parser.add_argument(
        "--store-fields",
        type=int,
        default=4096,
        help="Fields to read from the store; enough to sample from, cheap to load.",
    )
    parser.add_argument(
        "--min-field-peak",
        type=float,
        default=0.0,
        help="Drop whole fields whose absolute peak is below this fraction of the loudest.",
    )
    parser.add_argument(
        "--min-amplitude",
        type=float,
        default=0.0,
        help="Drop target coefficients below this fraction of their field's peak.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--json",
        type=pathlib.Path,
        default=None,
        help="Also write the measurement here, so a table can be rendered from it later.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the measurement and print the report."""
    args = parse_args()
    result = BridgeGeometryExperiment(
        pairs=args.pairs,
        coupling=args.coupling,
        seed=args.seed,
        ot_batch=args.ot_batch,
        target_store=args.target_store,
        store_fields=args.store_fields,
        min_amplitude=args.min_amplitude,
        min_field_peak=args.min_field_peak,
    ).run()
    print(render(result))

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "name": result.name,
                    "paper_reference": result.paper_reference,
                    "seed": result.seed,
                    "deterministic": result.deterministic,
                    "values": result.values,
                },
                indent=2,
            )
        )
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()

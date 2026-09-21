"""Table 1 -- analytical path geometry. No network, no data, ~20 s on a CPU.

The table has three target blocks. The synthetic one generates its own target and
reproduces from a bare checkout; the knee MRI and speech blocks read prebuilt
stores, and report MISSING INPUT with the command that builds them when they are
not here.

    uv run python -m reproducibility.table1_bridge_geometry
"""

from __future__ import annotations

import argparse
import pathlib

from cyfm.experiments.bridge_geometry import BridgeGeometryExperiment
from reproducibility._harness import (
    Check,
    MissingInput,
    compare,
    exit_with,
    missing,
    report,
)
from reproducibility.expected import TABLE1_SYNTHETIC, TABLE1_TAIL_INDEX

#: Stores the other two blocks of the table need, and how to build each.
STORES = {
    "Knee MRI": (
        pathlib.Path("data/knee_pd"),
        "uv run python scripts/data/build_knee_pd_store.py --help",
    ),
    "Speech STFT": (
        pathlib.Path("data/librispeech_stft"),
        "uv run python scripts/data/build_librispeech_stft_store.py --help",
    ),
}


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=int,
        default=1_000_000,
        help="Endpoint pairs; the paper uses 10^6 and so does the default.",
    )
    return parser.parse_args()


def main() -> int:
    """Reproduce the synthetic block and report on the other two."""
    args = parse_args()
    result = BridgeGeometryExperiment(pairs=args.pairs, coupling=0.5, seed=0).run()

    checks: list[Check] = []
    for arm, expectations in TABLE1_SYNTHETIC.items():
        measured = result.values["arms"][arm]
        for key, expectation in expectations.items():
            column = "paths > pi" if key == "exceeds_pi" else "energy > pi"
            checks.append(compare(f"Synthetic / {arm} / {column}", measured[key], expectation))

    checks.append(
        compare(
            "Cartesian tail index (Hill, top 1%)",
            result.values["tail_index"]["Cartesian / independent"],
            TABLE1_TAIL_INDEX,
        )
    )

    for block, (path, how) in STORES.items():
        if not path.exists():
            checks.append(
                missing(
                    f"{block} block",
                    MissingInput(f"{path} does not exist", how),
                )
            )

    return report(
        "TABLE 1 -- Analytical path geometry (Section 5.1)",
        checks,
        notes=[
            "The quantile columns the probe also prints are diagnostic: the peak has a "
            "Pareto tail of\nindex one, so an unweighted quantile of it moves with wherever "
            "the sample is cut. The paper\nprints the two shares above, which are bounded "
            "averages and need no threshold.",
        ],
    )


if __name__ == "__main__":
    exit_with(main())

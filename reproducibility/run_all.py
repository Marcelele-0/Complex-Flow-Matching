"""Run every reproduction script and report one verdict for the repository.

    uv run python -m reproducibility.run_all
    uv run python -m reproducibility.run_all --network-free   # only the fast ones
    uv run cyfm-reproduce                                     # the installed name

Three outcomes per script, and they mean different things. ``PASS`` is a number
that matches the paper. ``FAIL`` is a number that was computed and does not.
``MISSING INPUT`` is a number this repository cannot compute at all, because an
archive, a store or a checkpoint is not here; each of those says what to run to
produce it. The exit code is non-zero for either of the last two -- a script that
cannot find its input never prints a quiet "OK".
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass

__all__ = ["SCRIPTS", "main"]


@dataclass(frozen=True)
class Reproduction:
    """One reproduction script and what running it costs.

    Attributes:
        module: Module path, run with ``python -m``.
        title: What it reproduces.
        network_free: Whether it needs no trained checkpoint. These are the ones
            a reviewer can run on a laptop in minutes.
        runtime: Rough wall clock, for planning.
    """

    module: str
    title: str
    network_free: bool
    runtime: str


SCRIPTS: tuple[Reproduction, ...] = (
    Reproduction(
        "reproducibility.table1_bridge_geometry",
        "Table 1 -- analytical path geometry",
        network_free=True,
        runtime="~20 s",
    ),
    Reproduction(
        "reproducibility.table2_field_synthesis",
        "Table 2 -- spatial field synthesis (64x64)",
        network_free=True,
        runtime="~1 s",
    ),
    Reproduction(
        "reproducibility.table3_unet_scaling",
        "Table 3 -- dimensionality scaling",
        network_free=True,
        runtime="~1 s",
    ),
    Reproduction(
        "reproducibility.section53_ot_cost",
        "Section 5.3 -- OT cost saving against field size",
        network_free=True,
        runtime="~10 s",
    ),
    Reproduction(
        "reproducibility.table4_spiral",
        "Table 4 -- the Factorised Coupling Trap (spiral rows)",
        network_free=True,
        runtime="~4 min",
    ),
    Reproduction(
        "reproducibility.table4_patch_seams",
        "Table 4 / Section 5.4 -- patch seams",
        network_free=True,
        runtime="~40 s",
    ),
    Reproduction(
        "reproducibility.table5_knee_mri",
        "Table 5 -- knee MRI (64x64)",
        network_free=True,
        runtime="~1 s",
    ),
)


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network-free",
        action="store_true",
        help="Only the results that need no trained checkpoint.",
    )
    return parser.parse_args()


def main() -> int:
    """Run each script in turn and summarise.

    Returns:
        ``0`` only if every script reproduced everything it could.
    """
    args = parse_args()
    selected = [s for s in SCRIPTS if s.network_free or not args.network_free]

    outcomes: list[tuple[Reproduction, int]] = []
    for script in selected:
        print(f"\n>>> {script.module}  ({script.runtime})\n")
        completed = subprocess.run([sys.executable, "-m", script.module], check=False)
        outcomes.append((script, completed.returncode))

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    width = max(len(s.title) for s, _ in outcomes)
    for script, code in outcomes:
        print(f"  {'PASS' if code == 0 else 'NOT REPRODUCED':<16}{script.title:<{width}}")
    print("-" * 100)

    failed = [s for s, code in outcomes if code != 0]
    if failed:
        print(f"{len(failed)} of {len(outcomes)} results did not fully reproduce; see above.")
        return 1
    print(f"All {len(outcomes)} results reproduce the published numbers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

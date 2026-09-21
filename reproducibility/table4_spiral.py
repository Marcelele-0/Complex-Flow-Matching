"""Table 4 -- the Factorised Coupling Trap, spiral rows. ~4 min, CPU, no data.

Couples amplitude and phase independently on the pointwise copula target and
measures what survives. The paper's claim is that both marginals survive exactly
-- Kolmogorov-Smirnov distance zero -- while the dependence between them
collapses to about 0.04 at every rho, and the reported transport cost stays flat
at 0.20 while the true optimum rises to 0.38.

    uv run python -m reproducibility.table4_spiral
    uv run python -m reproducibility.table4_spiral --samples 256 --seeds 2   # quick

The defaults are the paper's: n = 1024 and 8 seeds, which the table's caption
states. Lowering either will fail the checks, and should.
"""

from __future__ import annotations

import argparse

from cyfm.data.synthetic import Structure
from cyfm.experiments.factorised_trap import FactorisedTrapExperiment
from reproducibility._harness import (
    Check,
    MissingInput,
    Outcome,
    compare,
    exit_with,
    missing,
    report,
)
from reproducibility.expected import TABLE4_SPIRAL

#: Typed as the Literal the toy accepts, so a rename there is caught here.
STRUCTURE: Structure = "spiral"


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1024, help="Endpoint pairs per draw.")
    parser.add_argument("--seeds", type=int, default=8, help="Repeats averaged over.")
    return parser.parse_args()


def main() -> int:
    """Measure the spiral rows and compare them with the paper."""
    args = parse_args()
    result = FactorisedTrapExperiment(
        samples=args.samples, seeds=args.seeds, structures=(STRUCTURE,)
    ).run()

    damage = {row["rho"]: row for row in result.values["damage"]}
    cost = {row["rho"]: row for row in result.values["cost"]}

    checks: list[Check] = []
    for rho, expectations in TABLE4_SPIRAL.items():
        if rho not in damage:
            checks.append(
                missing(
                    f"rho = {rho}",
                    MissingInput(
                        f"the sweep produced no row at rho = {rho}",
                        "check COUPLING_PRESETS in cyfm.data.synthetic",
                    ),
                )
            )
            continue
        checks.append(
            compare(
                f"rho {rho}  coupled r_cl", damage[rho]["rc"], expectations["coupled_correlation"]
            )
        )
        checks.append(
            compare(
                f"rho {rho}  cost factorised",
                cost[rho]["factorised"],
                expectations["cost_factorised"],
            )
        )
        checks.append(
            compare(f"rho {rho}  cost joint", cost[rho]["joint"], expectations["cost_joint"])
        )

    # The claim the numbers are evidence for, and the one every marginal
    # diagnostic misses: the marginals are untouched, so a test that looked only
    # at them would pass while the target was destroyed.
    worst_marginal = max(max(damage[rho]["ka"], damage[rho]["kp"]) for rho in damage)
    checks.append(
        Check(
            label="both marginals survive exactly (the trap)",
            outcome=Outcome.PASS if worst_marginal <= 1e-6 else Outcome.FAIL,
            measured=worst_marginal,
            expectation=None,
            detail=(
                f"worst Kolmogorov-Smirnov distance to the target marginals "
                f"{worst_marginal:.2e}; the paper reports 0"
            ),
        )
    )

    return report(
        "TABLE 4 -- the Factorised Coupling Trap (spiral rows)",
        checks,
        notes=[
            (
                f"n = {args.samples}, {args.seeds} seeds"
                + (
                    ", which the table's caption states."
                    if (args.samples, args.seeds) == (1024, 8)
                    else " -- NOT the paper's n = 1024 and 8 seeds, so these\n"
                    "numbers are not expected to match and the failures above say so."
                )
            ),
            "The cost the factorised plan reports is below the optimum because it is not\n"
            "transporting the target: it reassembles endpoints from different samples.",
        ],
    )


if __name__ == "__main__":
    exit_with(main())

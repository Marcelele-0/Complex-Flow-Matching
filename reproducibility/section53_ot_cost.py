"""Section 5.3 -- how much minibatch OT saves against field size. ~10 s, CPU, no data.

The paper quotes two points of this curve to explain why the knee results, which
are computed at 320x320, gain less from the coupling than the synthetic tables at
64x64: "transport cost by 2.8% rather than 0.5%".

    uv run python -m reproducibility.section53_ot_cost

The full sweep is printed beside the checks, because the two quoted points only
mean something against the shape of the curve they sit on -- and against the
`reordered` column, which stays near 100% at every size. The assignment does not
stop finding a permutation as dimension rises; every permutation simply costs
nearly the same.
"""

from __future__ import annotations

from cyfm.experiments.coupling_scaling import CouplingScalingExperiment, render as render_sweep
from reproducibility._harness import (
    Check,
    MissingInput,
    Outcome,
    compare,
    exit_with,
    missing,
    report,
)
from reproducibility.expected import SEC53_COST_DROP


def main() -> int:
    """Measure the cost-drop curve and check the two points the paper quotes."""
    result = CouplingScalingExperiment().run()
    by_side = {f"{row['side']}x{row['side']}": row for row in result.values["rows"]}

    checks: list[Check] = []
    for label, expectation in SEC53_COST_DROP.items():
        row = by_side.get(label)
        if row is None:
            checks.append(
                missing(
                    f"cost drop at {label}",
                    MissingInput(
                        f"the sweep did not include {label}",
                        "widen CouplingScalingExperiment(sides=...)",
                    ),
                )
            )
            continue
        checks.append(compare(f"cost drop at {label}", row["cost_drop"], expectation))

    # The claim the two numbers are evidence for, stated as its own check: the
    # saving falls monotonically as the field grows.
    drops = [row["cost_drop"] for row in result.values["rows"]]
    monotone = all(
        later <= earlier + 1e-9 for earlier, later in zip(drops, drops[1:], strict=False)
    )
    checks.append(
        Check(
            label="the saving falls monotonically with field size",
            outcome=Outcome.PASS if monotone else Outcome.FAIL,
            measured=drops[-1],
            expectation=None,
            detail=(
                f"{drops[0]:.1%} at 1x1 down to {drops[-1]:.1%} at 320x320, "
                f"{'monotone' if monotone else 'NOT monotone'}"
            ),
        )
    )

    return report(
        "SECTION 5.3 -- OT cost saving against field size",
        checks,
        notes=[render_sweep(result)],
    )


if __name__ == "__main__":
    exit_with(main())

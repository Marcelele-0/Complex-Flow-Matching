"""Table 3 -- dimensionality scaling, from the archived evaluations.

Seconds, CPU, no data and no checkpoint. Table 3's 64x64 block is Table 2's
rows, which is why that table's caption says it contains them; only the 16x16 and
32x32 blocks are checked here, so the two scripts do not both fail on one cause.
Averaging is done here rather than read from a stored aggregate, so what is
compared is the per-run evaluations themselves.

    uv run python -m reproducibility.table3_unet_scaling

This is a different question from the one ``tests/test_paper_results.py`` asks.
That test checks every archived run's Hydra overrides against the experiment
config it claims to come from -- that the runs are the ones the paper says they
are. This checks that the numbers in them are the numbers the paper prints.
"""

from __future__ import annotations

from reproducibility._harness import (
    Check,
    MissingInput,
    compare,
    exit_with,
    load_archive,
    mean_over_seeds,
    missing,
    report,
)
from reproducibility.expected import TABLE3_SLICED_W2, UNET_SEEDS

ARCHIVE = "unet_eval_metrics_l2u.json"
METRIC = "sliced_w2_complex"


def _label(arm: str) -> str:
    """Turn an archive run prefix into the row label the table prints.

    ``unsz_cylindrical_ot_16`` is the 16x16 CyFM joint-OT row; the table groups
    by resolution, so the side comes first.
    """
    _, geometry, coupling, side = arm.split("_", 3)
    name = "CyFM" if geometry == "cylindrical" else "Cartesian"
    # The table says "OT" for the Cartesian arms and "Joint OT" for CyFM.
    pairing = (
        ("joint OT" if geometry == "cylindrical" else "OT") if coupling == "ot" else "independent"
    )
    return f"{side}x{side} {name}, {pairing}"


def main() -> int:
    """Re-derive Table 2 from the archive and compare it with the paper."""
    checks: list[Check] = []
    notes: list[str] = []
    try:
        archive = load_archive(ARCHIVE)
    except MissingInput as error:
        return report(
            "TABLE 3 -- Dimensionality scaling",
            [missing("archived evaluations", error)],
        )

    for arm, expectations in TABLE3_SLICED_W2.items():
        label = _label(arm)
        for steps, expectation in expectations.items():
            try:
                mean, seeds = mean_over_seeds(archive, arm, METRIC, steps)
            except MissingInput as error:
                checks.append(missing(f"{label}, {steps} step(s)", error))
                continue
            check = compare(f"{label}, {steps:>3} step(s)", mean, expectation)
            checks.append(check)
            if seeds != UNET_SEEDS:
                notes.append(
                    f"{label} at {steps} step(s) averaged {seeds} seeds, not {UNET_SEEDS}: "
                    "the table's caption says five."
                )

    notes.append(
        "The archive is frozen input. tests/test_paper_results.py checks that each run's\n"
        "Hydra overrides match the experiment config it claims to come from; this script\n"
        "checks that the numbers inside those runs are the ones the paper prints."
    )
    return report("TABLE 3 -- Dimensionality scaling", checks, notes=notes)


if __name__ == "__main__":
    exit_with(main())

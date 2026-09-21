"""Table 5 -- physical field synthesis on fastMRI knee at 64x64, from the archive.

Seconds, CPU, no data and no checkpoint. Three blocks are checked: the sliced
W2 row for each of the four flow arms, the VE-SDE baseline that shares their
cohort and budget, and the spatial lag-1 gap -- the measure whose ordering the
paper reports as measured, including where the plane wins.

    uv run python -m reproducibility.table5_knee_mri

**Which archived runs.** Each arm appears twice, as ``dense_t5c64_*`` and
``p5_t5c64_*``. They score the same checkpoint on the same cohort with the same
seed and differ only in ``evaluate.nfe``: eleven step counts against five. Their
k=1..8 columns agree to float noise; their k=100 columns do not. The sweep
consumes one seeded generator across step counts, so a row depends on which step
counts preceded it -- k=100 is fifth in one list and eleventh in the other, and
draws a different prior batch. The table has five columns and its numbers are the
five-count evaluation, so that is what is checked here.
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
from reproducibility.expected import (
    TABLE5_ARMS,
    TABLE5_DIFFUSION,
    TABLE5_SLICED_W2,
    TABLE5_SPATIAL_LAG1,
    UNET_SEEDS,
)

ARCHIVE = "table5_fastmri64_metrics.json"
DIFFUSION_ARM = "t5c64_complex_diffusion_heun_independent"


def main() -> int:
    """Re-derive Table 5 from the archive and compare it with the paper."""
    try:
        archive = load_archive(ARCHIVE)
    except MissingInput as error:
        return report("TABLE 5 -- Knee MRI (64x64)", [missing("archived evaluations", error)])

    checks: list[Check] = []
    notes: list[str] = []

    def add(label: str, arm: str, metric: str, steps: int, expectation: object) -> None:
        try:
            mean, seeds = mean_over_seeds(archive, arm, metric, steps)
        except MissingInput as error:
            checks.append(missing(label, error))
            return
        checks.append(compare(label, mean, expectation))  # type: ignore[arg-type]
        if seeds != UNET_SEEDS:
            notes.append(f"{label} averaged {seeds} seeds, not {UNET_SEEDS}.")

    for arm, expectations in TABLE5_SLICED_W2.items():
        for steps, expectation in expectations.items():
            add(
                f"W2  {TABLE5_ARMS[arm]}, k={steps:>3}",
                arm,
                "sliced_w2_complex",
                steps,
                expectation,
            )

    for steps, expectation in TABLE5_DIFFUSION.items():
        add(
            f"W2  Complex diffusion (VE-SDE), k={steps:>3}",
            DIFFUSION_ARM,
            "sliced_w2_complex",
            steps,
            expectation,
        )

    for arm, expectations in TABLE5_SPATIAL_LAG1.items():
        for steps, expectation in expectations.items():
            add(
                f"lag1 {TABLE5_ARMS[arm]}, k={steps:>3}",
                arm,
                "spatial_lag1_gap",
                steps,
                expectation,
            )

    notes.append(
        "The score-based row's column spans four orders of magnitude: at k <= 2 its output\n"
        "is still the prior, whose scale is sigma_max = 53.4. Its tolerances follow the\n"
        "precision the table prints rather than one constant."
    )
    notes.append(
        "Checked against the five-count evaluation (p5_), which is what the table's five\n"
        "columns are. The eleven-count evaluation in the same archive agrees at k <= 8 and\n"
        "gives 0.0631 rather than 0.0722 at k = 100: the sweep shares one generator across\n"
        "step counts, so a row depends on which counts preceded it."
    )
    notes.append(
        "This archive records cluster paths in its provenance (/lustre/..., /mnt/lscratch/...).\n"
        "They travel with any public artefact; see reproducibility/README.md."
    )
    return report("TABLE 5 -- Knee MRI (64x64)", checks, notes=notes)


if __name__ == "__main__":
    exit_with(main())

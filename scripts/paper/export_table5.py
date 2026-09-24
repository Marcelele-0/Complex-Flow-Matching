"""Freeze the Table 5 evaluations as the archive the paper cites.

Table 5 is the one table this repository does not run locally: its grid is 25
training runs on 320x320 fastMRI knee volumes, launched by
``scripts/cluster/table5_fastmri.sbatch`` on the cluster. This script is the step
between that grid coming back and ``scripts/paper/latex_tables.py`` being able to
render anything -- it collects ``outputs/evaluate/t5_*_eval/*/metrics.json`` into
``docs/reproduce/paper_results/table5_fastmri_metrics.json``, keyed and shaped
exactly like the three archives ``paper_tables.py --export`` writes for Tables 2
and 3, and checked by ``tests/test_paper_results.py``.

Thirty evaluations from twenty-five training runs. The diffusion baseline is
scored twice from one checkpoint -- at the flow arms' matched budget and in its
own many-step regime -- so its two rows share a training directory, and the
many-step row is given that checkpoint's name explicitly rather than being left
with no training provenance at all.

Usage::

    uv run python scripts/paper/export_table5.py
    uv run python scripts/paper/export_table5.py --partial      # before the grid is whole
    uv run python scripts/paper/export_table5.py --if-present   # no-op off the cluster
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

from latex_tables import TABLE5_ARCHIVE, TABLE5_NAME, TABLE5_PREFIX, TABLE5_ROWS
from loss_protocols import RESULTS, SEEDS, Runs
from paper_tables import EVALUATIONS, display, provenance

ROOT = pathlib.Path(__file__).resolve().parents[2]

# The many-step row re-scores the matched row's checkpoint, so its training
# provenance is that run's. Keyed by arm, valued by the arm whose checkpoint it
# used; an arm absent from this mapping trained its own.
SCORED_FROM = {"complex_diffusion_native": "complex_diffusion_heun"}


def evaluation_names() -> list[tuple[str, str]]:
    """Every ``(evaluation name, training run name)`` pair Table 5 needs.

    Returns:
        Thirty pairs, in the row order of the table and seed-innermost. The evaluation
        name carries ``TABLE5_PREFIX``; the training name never does, because that
        re-scoring pass produced no checkpoints of its own. The two entries differ
        further only for the many-step row, which has no training run at all.
    """
    pairs = []
    for arm, coupling, _, _, _ in TABLE5_ROWS:
        trained = SCORED_FROM.get(arm, arm)
        for seed in SEEDS:
            pairs.append(
                (
                    TABLE5_NAME.format(prefix=TABLE5_PREFIX, arm=arm, coupling=coupling, seed=seed),
                    TABLE5_NAME.format(
                        prefix="", arm=trained, coupling=coupling, seed=seed
                    ).removesuffix("_eval"),
                )
            )
    return pairs


def load_outputs() -> tuple[Runs, list[str]]:
    """The newest ``metrics.json`` of every Table 5 evaluation present on disk.

    Returns:
        Tuple of the runs, keyed by evaluation name with their ``provenance``
        record attached, and the names that are missing.
    """
    runs: Runs = {}
    missing = []
    for name, trained in evaluation_names():
        found = sorted(EVALUATIONS.glob(f"{name}/*/metrics.json"))
        if not found:
            missing.append(name)
            continue
        run: dict[str, Any] = json.loads(found[-1].read_text())
        run["provenance"] = provenance(name, prefix="", train_name=trained)
        runs[name] = run
    return runs, missing


def export(path: pathlib.Path, partial: bool = False) -> None:
    """Write the archive, refusing an incomplete grid unless ``partial``.

    Args:
        path: Archive file to write.
        partial: Allow rows to be missing, for a spot check before the whole grid
            has come back.

    Raises:
        SystemExit: If rows are missing and ``partial`` is false, or nothing matched.
    """
    runs, missing = load_outputs()
    if not runs:
        raise SystemExit(
            f"No Table 5 evaluations under {display(EVALUATIONS)}. This grid runs on the "
            "cluster: see scripts/cluster/table5_fastmri.sbatch, then copy the evaluation "
            "directories back before exporting."
        )
    if missing and not partial:
        raise SystemExit(
            f"Refusing to export an incomplete archive; {len(missing)} of "
            f"{len(runs) + len(missing)} missing: {missing}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(runs, indent=1, sort_keys=True) + "\n")
    print(f"Wrote {len(runs)} evaluations to {display(path)}")
    if missing:
        print(f"PARTIAL: {len(missing)} still missing: {missing}")


def main() -> None:
    """Parse the command line and write the archive."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=RESULTS / TABLE5_ARCHIVE,
        help="Archive file to write.",
    )
    parser.add_argument(
        "--partial",
        action="store_true",
        help="Write an archive even though rows are missing.",
    )
    parser.add_argument(
        "--if-present",
        action="store_true",
        help=(
            "Exit 0 with a message when no evaluations are on this machine, so "
            "reproduce.py documents the step instead of failing off the cluster."
        ),
    )
    args = parser.parse_args()

    if args.if_present and not any(EVALUATIONS.glob("t5_*_eval/*/metrics.json")):
        print(
            "No Table 5 evaluations on this machine; skipping the export. "
            "The grid runs on the cluster: scripts/cluster/table5_fastmri.sbatch."
        )
        return
    export(args.out, partial=args.partial)


if __name__ == "__main__":
    main()

"""Freeze the speech evaluations behind the second block of Table 2.

The last place in the main text where a printed number had no source. The knee blocks
were archived on 2026-09-19 and 2026-09-20; the speech block was not, because
``conf/experiment/table6_audio.yaml`` carries no ``paper:`` block and so never entered
``reproduce.py``'s order. Twenty numbers and four k-star values were therefore in the
paper with nothing in the repository to regenerate them from.

**Which runs.** Three prefixes exist on the cluster and they are different experiments,
not retries: measured against the printed table, ``t6_`` and ``t6e10_`` miss every arm
(their cylindrical OT column reads 0.0423 and 0.0461 against the printed 0.0454), and
``t6e40_`` reproduces all four arms exactly. The launcher defaults to two epochs, which
is how the short cohorts came to exist, so the published one carries its epoch count in
its name.

Collector only, like ``export_table5c64.py``: the LaTeX is written by hand and this
makes every cell in it checkable against a frozen record.

Usage::

    uv run python scripts/paper/export_table6.py
    uv run python scripts/paper/export_table6.py --partial
    uv run python scripts/paper/export_table6.py --if-present
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

from latex_tables import TABLE6_ARCHIVE, TABLE6_NAME, TABLE6_PREFIX, TABLE6_ROWS
from loss_protocols import RESULTS, SEEDS, Runs
from paper_tables import EVALUATIONS, display, provenance


def evaluation_names() -> list[tuple[str, str]]:
    """Every ``(evaluation name, training run name)`` pair the speech block needs.

    Returns:
        Twenty pairs, in the row order of the table and seed-innermost. The evaluation
        name carries the prefix and the training name carries it too: unlike the knee
        cohorts, this one trained and scored under a single name.
    """
    pairs = []
    for arm, coupling, _, _, _ in TABLE6_ROWS:
        for seed in SEEDS:
            name = TABLE6_NAME.format(prefix=TABLE6_PREFIX, arm=arm, coupling=coupling, seed=seed)
            pairs.append((name, name.removesuffix("_eval")))
    return pairs


def load_outputs() -> tuple[Runs, list[str]]:
    """The newest ``metrics.json`` of every speech evaluation present on disk.

    Returns:
        Tuple of the runs, keyed by evaluation name with their ``provenance`` record
        attached, and the names that are missing.
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
        partial: Allow rows to be missing, for a spot check before the grid is back.

    Raises:
        SystemExit: If rows are missing and ``partial`` is false, or nothing matched.
    """
    runs, missing = load_outputs()
    if not runs:
        raise SystemExit(
            f"No speech evaluations under {display(EVALUATIONS)}. This grid runs on the "
            "cluster: see scripts/wcss/table6_audio.sbatch, which needs EPOCHS=40, then "
            "copy the evaluation directories back before exporting."
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
        "--out", type=pathlib.Path, default=RESULTS / TABLE6_ARCHIVE, help="Archive to write."
    )
    parser.add_argument(
        "--partial", action="store_true", help="Write an archive even though rows are missing."
    )
    parser.add_argument(
        "--if-present",
        action="store_true",
        help="Exit 0 with a message when no evaluations are on this machine.",
    )
    args = parser.parse_args()
    if args.if_present and not any(EVALUATIONS.glob(f"{TABLE6_PREFIX}*/*/metrics.json")):
        print(f"No speech evaluations under {display(EVALUATIONS)}; nothing to export.")
        return
    export(args.out, partial=args.partial)


if __name__ == "__main__":
    main()

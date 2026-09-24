"""Freeze the 64x64 knee evaluations as the archive the paper cites.

Table 5 of the main text is the one table in this repository whose numbers had no
archive at all: its ``.tex`` was written by hand, so until now no entry in it could
be regenerated from a checkout. That is the defect the prior control already cost an
evening over, on the paper's headline MRI result.

This is the collector, not the renderer. It reads the newest ``metrics.json`` of each
evaluation, attaches the Hydra overrides of that evaluation and of the training run it
scored, and writes ``docs/reproduce/paper_results/table5_fastmri64_metrics.json`` in the
shape ``paper_tables.py --export`` uses for every other archive. Rendering the table from
it is deliberately left out: the repository is due a restructure, and a renderer written
against the current module layout would be rewritten with it, while the archive is data
capture whose deadline is set by the cluster's scratch policy rather than by us.

**Newest stamp is resolved per evaluation directory, which is per arm and seed.** It
cannot be resolved per prefix: the diffusion arms were re-scored after the
sigma-preconditioning fix in two batches, one seed at ``2026-09-19_13:03`` and four at
``13:04``, so a single newest-stamp-for-the-whole-prefix rule would keep one seed and
silently drop four. The stamps before those carry the diverged runs (a many-step sliced
$W_2$ of 998 and 62164 against a data scale of 0.06) and must never be collected.

Usage::

    uv run python scripts/paper/export_table5c64.py
    uv run python scripts/paper/export_table5c64.py --partial      # before the grid is whole
    uv run python scripts/paper/export_table5c64.py --if-present   # no-op off the cluster
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

from latex_tables import (
    TABLE5C64_ARCHIVE,
    TABLE5C64_DENSE_ARMS,
    TABLE5C64_DENSE_PREFIX,
    TABLE5C64_NAME,
    TABLE5C64_ROWS,
)
from loss_protocols import RESULTS, SEEDS, Runs
from paper_tables import EVALUATIONS, display, provenance

# The many-step diffusion row re-scores the matched row's checkpoint, so its training
# provenance is that run's. Same convention as export_table5.py: an arm absent from this
# mapping trained its own. Without it that row's train_* keys would be silently absent
# and the row unverifiable.
SCORED_FROM = {"complex_diffusion_native": "complex_diffusion_heun"}


def evaluation_names() -> list[tuple[str, str]]:
    """Every ``(evaluation name, training run name)`` pair the 64x64 block needs.

    Returns:
        Fifty pairs, seed-innermost: the six printed rows, then the four flow arms
        again on the dense grid the caption's k* is read off. The evaluation name
        carries its prefix; the training name never does, because neither sweep
        produced checkpoints of its own -- both scored the same twenty-five runs.
    """
    wanted = [(arm, coupling, prefix) for arm, coupling, prefix, _, _, _ in TABLE5C64_ROWS]
    wanted += [(arm, coupling, TABLE5C64_DENSE_PREFIX) for arm, coupling in TABLE5C64_DENSE_ARMS]

    pairs = []
    for arm, coupling, prefix in wanted:
        trained = SCORED_FROM.get(arm, arm)
        for seed in SEEDS:
            pairs.append(
                (
                    TABLE5C64_NAME.format(prefix=prefix, arm=arm, coupling=coupling, seed=seed),
                    TABLE5C64_NAME.format(
                        prefix="", arm=trained, coupling=coupling, seed=seed
                    ).removesuffix("_eval"),
                )
            )
    return pairs


def load_outputs() -> tuple[Runs, list[str]]:
    """The newest ``metrics.json`` of every 64x64 knee evaluation present on disk.

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
        # prefix="" because `name` already carries it; the training directory never
        # does, so train_name is passed unprefixed.
        run["provenance"] = provenance(name, prefix="", train_name=trained)
        runs[name] = run
    return runs, missing


def export(path: pathlib.Path, partial: bool = False) -> None:
    """Write the archive, refusing an incomplete grid unless ``partial``.

    Args:
        path: Archive file to write.
        partial: Allow rows to be missing, for a spot check before the whole grid is back.

    Raises:
        SystemExit: If rows are missing and ``partial`` is false, or nothing matched.
    """
    runs, missing = load_outputs()
    if not runs:
        raise SystemExit(
            f"No 64x64 knee evaluations under {display(EVALUATIONS)}. This grid runs on "
            "the cluster: see scripts/cluster/table5_fastmri.sbatch with KSPACE_CROP=64, then "
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
        "--out",
        type=pathlib.Path,
        default=RESULTS / TABLE5C64_ARCHIVE,
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
    if args.if_present and not any(EVALUATIONS.glob("*t5c64*/*/metrics.json")):
        print(f"No 64x64 knee evaluations under {display(EVALUATIONS)}; nothing to export.")
        return
    export(args.out, partial=args.partial)


if __name__ == "__main__":
    main()

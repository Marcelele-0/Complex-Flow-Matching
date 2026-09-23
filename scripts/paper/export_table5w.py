r"""Freeze the amplitude-weighted knee arms, which one sentence in Section 5.2 rests on.

That sentence says the unweighted product metric $\\mathrm{d}A^2 + \\mathrm{d}\\theta^2$
beats an amplitude-weighted phase term on every spatial measure. The comparison is
0.0153 against 0.0219 on amplitude texture and 0.1057 against 0.1656 on phase coherence,
both at $k=100$ -- and until now the weighted half of it was in no archive at all, so the
claim had no source anywhere in the repository while everything around it did.

The runs are ``t5wc64_``: the weighted cohort at the 64x64 acquisition matrix. As with
every other block, the published numbers come from the ``p5_`` re-scoring pass, because
the unprefixed runs predate the phase-coherence metric and carry no ``phase_lag1_gap``.
Measured against the printed sentence, ``p5_`` reproduces both values exactly and the
unprefixed pass reproduces neither (0.0226 and 0.2387).

Usage::

    uv run python scripts/paper/export_table5w.py
    uv run python scripts/paper/export_table5w.py --partial
    uv run python scripts/paper/export_table5w.py --if-present
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

from latex_tables import TABLE5W_ARCHIVE, TABLE5W_NAME, TABLE5W_PREFIX, TABLE5W_ROWS
from loss_protocols import RESULTS, SEEDS, Runs
from paper_tables import EVALUATIONS, display, provenance


def evaluation_names() -> list[tuple[str, str]]:
    """Every ``(evaluation name, training run name)`` pair the weighted block needs.

    Returns:
        Twenty pairs, seed-innermost. The evaluation carries the re-scoring prefix; the
        training run does not, since that pass produced no checkpoints of its own.
    """
    pairs = []
    for arm, coupling in TABLE5W_ROWS:
        for seed in SEEDS:
            pairs.append(
                (
                    TABLE5W_NAME.format(
                        prefix=TABLE5W_PREFIX, arm=arm, coupling=coupling, seed=seed
                    ),
                    TABLE5W_NAME.format(
                        prefix="", arm=arm, coupling=coupling, seed=seed
                    ).removesuffix("_eval"),
                )
            )
    return pairs


def load_outputs() -> tuple[Runs, list[str]]:
    """The newest ``metrics.json`` of every weighted evaluation present on disk.

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
        partial: Allow rows to be missing.

    Raises:
        SystemExit: If rows are missing and ``partial`` is false, or nothing matched.
    """
    runs, missing = load_outputs()
    if not runs:
        raise SystemExit(
            f"No weighted knee evaluations under {display(EVALUATIONS)}. They run on the "
            "cluster with training.loss.phase_amplitude_weighting=true and "
            "dataset.kspace_crop=64; copy the evaluation directories back first."
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
        "--out", type=pathlib.Path, default=RESULTS / TABLE5W_ARCHIVE, help="Archive to write."
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
    if args.if_present and not any(EVALUATIONS.glob(f"{TABLE5W_PREFIX}t5wc64_*/*/metrics.json")):
        print(f"No weighted knee evaluations under {display(EVALUATIONS)}; nothing to export.")
        return
    export(args.out, partial=args.partial)


if __name__ == "__main__":
    main()

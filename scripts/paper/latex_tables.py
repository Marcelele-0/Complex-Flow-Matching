"""Write the LaTeX of Tables 2 and 3 and the appendix loss table from the archives.

The paper includes these files with ``\\input``, so every number in the tables is
read from ``docs/reproduce/paper_results/`` rather than typed by hand:

* ``table2_unet64.tex`` -- 64x64, both geometries x both couplings, matched loss;
* ``table3_unet_scaling.tex`` -- 16/32/64, Cartesian OT and the cylinder with and
  without the joint coupling, matched loss;
* ``tableA_loss_protocols.tex`` -- every arm under both losses (24 rows).

Bold marks the lowest mean in a column of Tables 2 and 3 only when all its seeds lie
below all seeds of the best arm of the other geometry; otherwise, as at every
k = 100, nothing is bold. The captions' statistics (the p-values and the exception
cells) are computed here as well, with the tests of ``loss_protocols.py``.

Usage::

    uv run python scripts/paper/latex_tables.py --out docs/reproduce/paper_results/latex
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from loss_protocols import (
    COUPLINGS,
    L1_LOSS,
    RESULTS,
    SEEDS,
    SIDES,
    STEPS,
    Runs,
    best,
    load,
    values,
)
from scipy.stats import mannwhitneyu

ROOT = pathlib.Path(__file__).resolve().parents[2]
MATCHED = "l2u"
METRIC = "sliced_w2_complex"
GEOMETRY_LABELS = {"euclidean": r"Cartesian ($\mathbb{R}^2$)", "cylindrical": "Cylindrical (Ours)"}
STEP_HEADER = " & ".join(rf"\textbf{{{k} Step{'s' if k > 1 else ''}}}" for k in STEPS)
RESOLUTION_HEADER = r"\textbf{Resolution} & \textbf{Geometry} & \textbf{Coupling}"


def separated(low: np.ndarray, high: np.ndarray) -> bool:
    """True when every seed of ``low`` lies strictly below every seed of ``high``."""
    return bool(low.max() < high.min())


def p_value(a: np.ndarray, b: np.ndarray) -> float:
    """Exact two-sided Mann-Whitney p between two sets of per-seed values."""
    return float(mannwhitneyu(a, b, alternative="two-sided", method="exact").pvalue)


def cell(value: float, bold: bool) -> str:
    """One table entry, three decimals, optionally bold."""
    text = f"{value:.3f}"
    return rf"\textbf{{{text}}}" if bold else text


def column_bold(
    sources: dict[str, Runs],
    side: int,
    step: int,
    rows: list[tuple[str, str]],
) -> int | None:
    """The row to bold in one column, if its advantage over the other geometry holds.

    Args:
        sources: Loaded archives, keyed by loss label.
        side: Field side of the block.
        step: Solver step count of the column.
        rows: The block's (geometry, coupling) rows, all under the matched loss.

    Returns:
        Index of the lowest-mean row when all its seeds are below all seeds of the
        other geometry's best row in this block, else ``None``.
    """
    arms = [values(sources, MATCHED, side, g, c, step, METRIC) for g, c in rows]
    winner = int(np.argmin([arm.mean() for arm in arms]))
    rivals = [i for i, (g, _) in enumerate(rows) if g != rows[winner][0]]
    rival = min(rivals, key=lambda i: float(arms[i].mean()))
    return winner if separated(arms[winner], arms[rival]) else None


def matched_convergence_p(sources: dict[str, Runs], side: int) -> float:
    """p at k = 100 between cylinder + OT and the better Cartesian coupling (matched loss)."""
    step = STEPS[-1]
    cylinder = values(sources, MATCHED, side, "cylindrical", "ot", step, METRIC)
    _, cartesian = best(sources, [(MATCHED, c) for c in COUPLINGS], side, "euclidean", step, METRIC)
    return p_value(cylinder, cartesian)


def table2(sources: dict[str, Runs]) -> str:
    """Table 2: 64x64, both geometries x both couplings, matched loss."""
    side = 64
    rows = [(g, c) for g in ("euclidean", "cylindrical") for c in COUPLINGS]
    bold = {step: column_bold(sources, side, step, rows) for step in STEPS}
    lines = []
    for index, (geometry, coupling) in enumerate(rows):
        entries = []
        for step in STEPS:
            arm = values(sources, MATCHED, side, geometry, coupling, step, METRIC)
            entries.append(cell(float(arm.mean()), bold[step] == index))
        label = GEOMETRY_LABELS[geometry]
        coupling_label = "Independent" if coupling == "independent" else "Minibatch OT"
        if geometry == "cylindrical":
            label, coupling_label = rf"\textbf{{{label}}}", rf"\textbf{{{coupling_label}}}"
        lines.append(f"        {label} & {coupling_label} & " + " & ".join(entries) + r" \\")
        if index == 1:
            lines.append(r"        \midrule")
    p_conv = matched_convergence_p(sources, side)
    caption = (
        r"\textbf{Spatial Field Synthesis ($64\times64$).} Generative error (Sliced $W_2$, "
        r"$\downarrow$), mean over 5 seeds. All arms are trained with the same objective, the "
        r"unweighted squared error of the velocity. CyFM with joint OT has the lowest error up "
        r"to $k = 8$, with all seeds separated; at $k = 100$ no arm is distinguishable "
        rf"($p = {p_conv:.2f}$). Appendix Table~\ref{{tab:unet_loss_protocols}} repeats the "
        r"comparison with each geometry's best loss."
    )
    return "\n".join(
        [
            r"\begin{table}[h]",
            r"    \centering",
            rf"    \caption{{{caption}}}",
            r"    \label{tab:unet_baseline}",
            r"    \vspace{1mm}",
            r"    \begin{tabular}{llccccc}",
            r"        \toprule",
            rf"        \textbf{{Geometry}} & \textbf{{Coupling}} & {STEP_HEADER} \\",
            r"        \midrule",
            *lines,
            r"        \bottomrule",
            r"    \end{tabular}",
            r"    \vspace{1mm}",
            r"\end{table}",
            "",
        ]
    )


def table3(sources: dict[str, Runs]) -> str:
    """Table 3: every resolution, Cartesian OT and the cylinder with and without OT."""
    rows = [("euclidean", "ot"), ("cylindrical", "independent"), ("cylindrical", "ot")]
    labels = {
        ("euclidean", "ot"): ("Cartesian", "OT"),
        ("cylindrical", "independent"): ("CyFM", "Independent"),
        ("cylindrical", "ot"): (r"\textbf{CyFM (Ours)}", r"\textbf{Joint OT}"),
    }
    blocks = []
    for side in SIDES:
        bold = {step: column_bold(sources, side, step, rows) for step in STEPS}
        block = [f"        ${side}\\times{side}$ ($D={side * side}$)"]
        for index, arm in enumerate(rows):
            entries = [
                cell(
                    float(values(sources, MATCHED, side, *arm, step, METRIC).mean()),
                    bold[step] == index,
                )
                for step in STEPS
            ]
            geometry, coupling = labels[arm]
            block.append(f"        & {geometry} & {coupling} & " + " & ".join(entries) + r" \\")
        blocks.append("\n".join(block))
    caption = (
        r"\textbf{Dimensionality Scaling.} Generative error (Sliced $W_2$, $\downarrow$) "
        r"across spatial resolutions, mean over 5 seeds, with the cylindrical model trained "
        r"with and without the joint OT coupling. All arms use the unweighted squared velocity "
        r"error. CyFM's few-step advantage holds at all three resolutions, and joint OT "
        r"improves the cylindrical model at every step count."
    )
    return "\n".join(
        [
            r"\begin{table}[h]",
            r"    \centering",
            rf"    \caption{{{caption}}}",
            r"    \label{tab:unet_scaling}",
            r"    \vspace{1mm}",
            r"    \resizebox{\linewidth}{!}{",
            r"    \begin{tabular}{lllccccc}",
            r"        \toprule",
            rf"        {RESOLUTION_HEADER} & {STEP_HEADER} \\",
            r"        \midrule",
            ("\n" + r"        \midrule" + "\n").join(blocks),
            r"        \bottomrule",
            r"    \end{tabular}",
            r"    }",
            r"    \vspace{1mm}",
            r"\end{table}",
            "",
        ]
    )


def best_loss_statistics(sources: dict[str, Runs]) -> tuple[list[str], float]:
    """Exceptions to the few-step advantage and the smallest p at convergence.

    Both geometries take, per cell, their lowest-mean variant of {L1, L2} x {couplings}.

    Returns:
        The ``side x side, k = step (p = ...)`` cells with k <= 8 whose seeds do not
        separate in the cylinder's favour, and the smallest p over the k = 100 cells.
    """
    cylinder_arms = [(loss, c) for loss in (MATCHED, L1_LOSS["cylindrical"]) for c in COUPLINGS]
    cartesian_arms = [(loss, c) for loss in (MATCHED, L1_LOSS["euclidean"]) for c in COUPLINGS]
    exceptions, convergence = [], []
    for side in SIDES:
        for step in STEPS:
            _, cyl = best(sources, cylinder_arms, side, "cylindrical", step, METRIC)
            _, cart = best(sources, cartesian_arms, side, "euclidean", step, METRIC)
            p = p_value(cyl, cart)
            if step == STEPS[-1]:
                convergence.append(p)
            elif not separated(cyl, cart):
                exceptions.append(rf"${side}\times{side}$, $k = {step}$ ($p = {p:.2f}$)")
    return exceptions, min(convergence)


def table_appendix(sources: dict[str, Runs]) -> str:
    """Appendix table: both geometries x both couplings x both losses, every resolution."""
    blocks = []
    for side in SIDES:
        block = []
        first = True
        for geometry in ("euclidean", "cylindrical"):
            for coupling in COUPLINGS:
                for loss_label, loss in (("L1", L1_LOSS[geometry]), ("L2", MATCHED)):
                    means = [
                        float(values(sources, loss, side, geometry, coupling, step, METRIC).mean())
                        for step in STEPS
                    ]
                    entries = [f"{mean:.3f}" for mean in means]
                    resolution = rf"${side}\times{side}$" if first else ""
                    first = False
                    name = "Cartesian" if geometry == "euclidean" else "CyFM"
                    coupling_label = (
                        "Independent"
                        if coupling == "independent"
                        else ("OT" if geometry == "euclidean" else "Joint OT")
                    )
                    block.append(
                        f"        {resolution} & {name} & {coupling_label} & {loss_label} & "
                        + " & ".join(entries)
                        + r" \\"
                    )
        blocks.append("\n".join(block))
    exceptions, p_conv = best_loss_statistics(sources)
    exception_text = "except " + "; ".join(exceptions) if exceptions else "without exception"
    caption = (
        r"\textbf{Both geometries under both losses.} Generative error (Sliced $W_2$, "
        r"$\downarrow$), mean over 5 seeds. L1 / L2: absolute / squared error of the velocity; "
        r"the cylindrical phase term is unweighted in both. Taking each geometry's best of the "
        r"four variants per cell, CyFM has the lower error at every $k \le 8$ with all seeds "
        rf"separated, {exception_text}; at $k = 100$ no best-against-best comparison is "
        rf"significant ($p \ge {np.floor(p_conv * 100) / 100:.2f}$)."
    )
    return "\n".join(
        [
            r"\begin{table}[h]",
            r"    \centering",
            rf"    \caption{{{caption}}}",
            r"    \label{tab:unet_loss_protocols}",
            r"    \vspace{1mm}",
            r"    \resizebox{\linewidth}{!}{",
            r"    \begin{tabular}{llllccccc}",
            r"        \toprule",
            rf"        {RESOLUTION_HEADER} & \textbf{{Loss}} & {STEP_HEADER} \\",
            r"        \midrule",
            ("\n" + r"        \midrule" + "\n").join(blocks),
            r"        \bottomrule",
            r"    \end{tabular}",
            r"    }",
            r"    \vspace{1mm}",
            r"\end{table}",
            "",
        ]
    )


# --- Table 5: fastMRI knee CORPD, real complex data ---

# The archive scripts/paper/export_table5.py writes for this table, and the
# run-name convention of scripts/wcss/table5_fastmri.sbatch. Kept beside each
# other because a mismatch between them is silent: a missing seed raises, but a
# wrong prefix produces an empty table. The `_eval` suffix is part of the key:
# every archive in this repository is keyed by the *evaluation* directory name,
# which run_arm.sh writes as "{run}_eval".
TABLE5_ARCHIVE = "table5_fastmri_metrics.json"
TABLE5_NAME = "{prefix}t5_{arm}_{coupling}_s{seed}_eval"

# The evaluations Appendix D is printed from. The unprefixed `t5_*` directories are the
# original scoring pass and are superseded: they predate the phase-coherence metric and
# carry no `phase_lag1_gap` key at all, so half that table cannot be built from them, and
# their sliced W2 differs in the third decimal (cylindrical OT at k=1 is 0.0565 there
# against the printed 0.0561). Collecting them produced an archive that disagreed with
# the paper on every row, which is the whole failure this archive exists to prevent.
TABLE5_PREFIX = "p5_"

# The step count the diffusion baseline's own regime is reported at. Score-based
# samplers are used in the hundreds-to-thousands, not at the handful of steps the
# matched-NFE columns allow, so this row is swept on its own and occupies the last
# column alone -- see TABLE5_ROWS.
TABLE5_NATIVE_STEPS = 1000

# Six lines, five arms: the diffusion baseline appears twice because the same
# checkpoint is scored under two step budgets, and reporting only the matched one
# would hold it to a regime it is not used in while reporting only its own would
# not answer the table's question.
#
# The fourth field is the step counts that row was swept at. Every matched-cost
# row carries STEPS; the many-step row carries its own single point, because the
# columns of this table are matched *budgets* and a 1000-step sample does not sit
# in one. Its other cells print a dash rather than a number from the wrong regime.
TABLE5_ROWS = (
    ("cylindrical", "ot", "Cylindrical (Ours)", "Minibatch OT", STEPS),
    ("cylindrical", "independent", "Cylindrical (Ours)", "Independent", STEPS),
    ("euclidean", "ot", r"Cartesian ($\mathbb{R}^2$)", "Minibatch OT", STEPS),
    ("euclidean", "independent", r"Cartesian ($\mathbb{R}^2$)", "Independent", STEPS),
    ("complex_diffusion_heun", "independent", "Complex Diffusion", "Matched NFE", STEPS),
    (
        "complex_diffusion_native",
        "independent",
        "Complex Diffusion",
        rf"{TABLE5_NATIVE_STEPS} steps$^\dagger$",
        (TABLE5_NATIVE_STEPS,),
    ),
)

# Rows the bolding rule may consider. The many-step row is not a matched-cost
# comparison, so it can neither win a column nor block another row from winning
# one; leaving it in would let a number from a different budget decide the bold.
TABLE5_MATCHED_ROWS = tuple(index for index, row in enumerate(TABLE5_ROWS) if row[4] == STEPS)

# The 64x64 knee block -- Table 5 of the main text, where the 320x320 grid above is
# Appendix D. It is drawn from TWO sweeps of the same checkpoints, and which number
# came from which is not recoverable from the printed table, so it is pinned here.
#
# `cfm.evaluate` consumes one seeded generator sequentially down its NFE list, so a
# step count draws a different prior depending on where it sits in the list. The two
# sweeps share the prefix 1, 2, 4, 8 and those columns are byte-identical between
# them; k=100 sits fifth in the short list and eleventh in the dense one, and there
# the cylindrical OT mean is 0.0722 against 0.0631. Both are honest estimates over 64
# fields -- the dense sweep's own interior is non-monotone by more than that spread --
# but they are different draws and a table must not mix them across its columns.
#
# So: the printed columns are the five-point sweep (`p5_` prefix), and the dense
# eleven-point sweep (`dense_`) is what the caption's k* is read off. k* is 1, 1, 8, 8
# on both grids, which is why the two can coexist in one table; the archive keeps both
# so that stays checkable rather than remembered.
TABLE5C64_ARCHIVE = "table5_fastmri64_metrics.json"
TABLE5C64_NAME = "{prefix}t5c64_{arm}_{coupling}_s{seed}_eval"
TABLE5C64_DENSE_STEPS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 100)
TABLE5C64_DENSE_PREFIX = "dense_"

# Six lines, five arms, as in TABLE5_ROWS: the diffusion baseline is scored twice from
# one checkpoint. The third field is the evaluation-name prefix -- the diffusion rows
# were never re-swept and carry none -- and the sixth the step counts of that sweep.
TABLE5C64_ROWS = (
    ("cylindrical", "ot", "p5_", "Cylindrical (Ours)", "Minibatch OT", STEPS),
    ("cylindrical", "independent", "p5_", "Cylindrical (Ours)", "Independent", STEPS),
    ("euclidean", "ot", "p5_", r"Cartesian ($\mathbb{R}^2$)", "Minibatch OT", STEPS),
    ("euclidean", "independent", "p5_", r"Cartesian ($\mathbb{R}^2$)", "Independent", STEPS),
    ("complex_diffusion_heun", "independent", "", "Complex Diffusion", "Matched NFE", STEPS),
    (
        "complex_diffusion_native",
        "independent",
        "",
        "Complex Diffusion",
        rf"{TABLE5_NATIVE_STEPS} steps$^\dagger$",
        (TABLE5_NATIVE_STEPS,),
    ),
)

# The flow arms again, on the dense grid, for the caption's k*. Same checkpoints and
# same protocol as the rows above; only the NFE list differs.
TABLE5C64_DENSE_ARMS = tuple(
    (arm, coupling) for arm, coupling, prefix, _, _, _ in TABLE5C64_ROWS if prefix == "p5_"
)


def load_table5() -> Runs:
    """The Table 5 archive: run name to that run's ``metrics.json`` payload."""
    return dict(json.loads((RESULTS / TABLE5_ARCHIVE).read_text()))


def table5_values(runs: Runs, arm: str, coupling: str, step: int, metric: str) -> np.ndarray:
    """Per-seed values of one arm at one step count; raises if a seed is missing."""
    out = []
    for seed in SEEDS:
        run = runs[TABLE5_NAME.format(prefix=TABLE5_PREFIX, arm=arm, coupling=coupling, seed=seed)]
        out.append(next(row[metric] for row in run["sweep"] if row["num_steps"] == step))
    return np.array(out, dtype=float)


def table5(runs: Runs) -> str:
    """Table 5: fastMRI knee CORPD at 320x320, the arms at matched cost.

    The primary metric is the one the other tables already report, fixed before
    any of these numbers existed; the secondary metrics of the issue's hierarchy
    live in the archive and are read from there, not bolded here.
    """
    # None where a row was not swept at that column's budget, so a cell can be a
    # dash rather than a number borrowed from another regime.
    entries_per_row: list[list[float | None]] = []
    for arm, coupling, _, _, swept in TABLE5_ROWS:
        entries_per_row.append(
            [
                float(table5_values(runs, arm, coupling, step, METRIC).mean())
                if step in swept
                else None
                for step in STEPS
            ]
        )
    # The many-step row is swept at one point and shown in the last column.
    native = len(TABLE5_ROWS) - 1
    entries_per_row[native][-1] = float(
        table5_values(
            runs, TABLE5_ROWS[native][0], TABLE5_ROWS[native][1], TABLE5_NATIVE_STEPS, METRIC
        ).mean()
    )

    # Bold only where every seed of the leader sits below every seed of the best
    # arm of another family -- the rule column_bold applies: separation is tested
    # against the lowest-mean rival, not against every rival at once.
    bold: dict[int, int] = {}
    for column, step in enumerate(STEPS):
        # Bound to the value rather than looked up inside the key function: a row is a
        # candidate exactly when its cell is not a dash, so the mapping is total over
        # `candidates` and neither `min` below can be handed a None to order.
        column_values = {
            index: value
            for index in TABLE5_MATCHED_ROWS
            if (value := entries_per_row[index][column]) is not None
        }
        candidates = list(column_values)
        leader = min(candidates, key=lambda index: column_values[index])
        leader_family = TABLE5_ROWS[leader][0].split("_")[0]
        low = table5_values(runs, TABLE5_ROWS[leader][0], TABLE5_ROWS[leader][1], step, METRIC)
        rivals = [
            index for index in candidates if TABLE5_ROWS[index][0].split("_")[0] != leader_family
        ]
        if not rivals:
            continue
        closest = min(rivals, key=lambda index: column_values[index])
        rival = table5_values(runs, TABLE5_ROWS[closest][0], TABLE5_ROWS[closest][1], step, METRIC)
        if separated(low, rival):
            bold[step] = leader

    lines = []
    for index, (_, _, label, coupling_label, _) in enumerate(TABLE5_ROWS):
        cells = [
            r"--" if value is None else cell(value, bold.get(step) == index)
            for value, step in zip(entries_per_row[index], STEPS, strict=True)
        ]
        shown = rf"\textbf{{{label}}}" if label.startswith("Cylindrical") else label
        lines.append(f"        {shown} & {coupling_label} & " + " & ".join(cells) + r" \\")
        if index in (1, 3):
            lines.append(r"        \midrule")

    caption = (
        r"\textbf{Complex-Valued MRI Synthesis (fastMRI knee CORPD, $320\times320$).} "
        r"Generative error (Sliced $W_2$, $\downarrow$), mean over 5 seeds, on real "
        r"complex data rather than the synthetic copula of Tables 2 and 3. Every arm "
        r"is trained with the same objective, schedule and batch size, and scored "
        r"against volumes no arm was trained on. Step counts are model evaluations "
        r"matched across arms: Heun spends $2k-1$ calls, and the diffusion baseline "
        r"is reported both within that budget and, separately, in the many-step "
        r"regime it is actually used in. "
        r"$^\dagger$The last row is a second scoring of the matched row's "
        rf"checkpoint at {TABLE5_NATIVE_STEPS} steps "
        rf"(${2 * TABLE5_NATIVE_STEPS}$ evaluations); it is not a matched-cost "
        r"column, so its other entries are left blank and it is excluded from the "
        r"comparison the bold marks. "
        r"The transport-cost saving of minibatch OT falls from 86\% for scalars to "
        r"3\% at $64\times64$, so at this resolution the coupling may buy nothing "
        r"measurable; the geometry claim does not rest on it."
    )
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"    \centering",
            r"    \caption{" + caption + "}",
            r"    \label{tab:fastmri_knee}",
            r"    \begin{tabular}{ll" + "c" * len(STEPS) + "}",
            r"        \toprule",
            r"        \textbf{Geometry} & \textbf{Coupling} & " + STEP_HEADER + r" \\",
            r"        \midrule",
            *lines,
            r"        \bottomrule",
            r"    \end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def main() -> None:
    """Write the table files."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=ROOT / "docs" / "reproduce" / "paper_results" / "latex",
        help="Directory for the .tex files.",
    )
    args = parser.parse_args()

    sources = load(archive=True)
    args.out.mkdir(parents=True, exist_ok=True)
    for filename, text in (
        ("table2_unet64.tex", table2(sources)),
        ("table3_unet_scaling.tex", table3(sources)),
        ("tableA_loss_protocols.tex", table_appendix(sources)),
    ):
        (args.out / filename).write_text(text)
        print(f"wrote {args.out / filename}")

    # Written only once its grid has been run and archived WHOLE. Existence of the file
    # is not enough: the archive is legitimately partial, because the score-based
    # baseline was run at 64x64 only and Appendix D has no diffusion row, so the two
    # diffusion lines this renderer models have no evaluations at 320. Rendering anyway
    # would raise a KeyError from inside the bolding loop rather than say which rows are
    # missing, and the paper does not \input this file -- Appendix D's table is written
    # inline against the same archive.
    if not (RESULTS / TABLE5_ARCHIVE).exists():
        print(f"skipped table5_fastmri.tex: no {RESULTS / TABLE5_ARCHIVE} yet")
        return
    runs = load_table5()
    absent = [
        name
        for arm, coupling, _, _, _ in TABLE5_ROWS
        for seed in SEEDS
        if (name := TABLE5_NAME.format(prefix=TABLE5_PREFIX, arm=arm, coupling=coupling, seed=seed))
        not in runs
    ]
    if absent:
        print(f"skipped table5_fastmri.tex: archive is partial, {len(absent)} rows missing")
        return
    destination = args.out / "table5_fastmri.tex"
    destination.write_text(table5(runs))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()

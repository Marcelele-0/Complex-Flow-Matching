"""Write the LaTeX of Tables 2 and 3 and the appendix loss table from the archives.

The paper includes these files with ``\\input``, so every number in the tables is
read from ``docs/paper_results/`` rather than typed by hand:

* ``table2_unet64.tex`` -- 64x64, both geometries x both couplings, matched loss;
* ``table3_unet_scaling.tex`` -- 16/32/64, Cartesian OT and the cylinder with and
  without the joint coupling, matched loss;
* ``tableA_loss_protocols.tex`` -- every arm under both losses (24 rows).

Bold marks the lowest mean in a column of Tables 2 and 3 only when all its seeds lie
below all seeds of the best arm of the other geometry; otherwise, as at every
k = 100, nothing is bold. The captions' statistics (the p-values and the exception
cells) are computed here as well, with the tests of ``loss_protocols.py``.

Usage::

    uv run python scripts/paper/latex_tables.py --out docs/paper_results/latex
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
from loss_protocols import COUPLINGS, L1_LOSS, SIDES, STEPS, Runs, best, load, values
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


def main() -> None:
    """Write the three table files."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=ROOT / "docs" / "paper_results" / "latex",
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


if __name__ == "__main__":
    main()

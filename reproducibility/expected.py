"""The numbers the paper prints, as constants, with the tolerance each deserves.

Every value here is quoted from ``paper/ICLR Main/``, with the file and the
sentence it comes from named beside it. Nothing is copied from a script's output
or from a ``conf/experiment/*.yaml`` ``expected:`` block: those record what the
code produced on some day, which is the thing being checked, not the claim.

Tolerances are per-number and argued, not global. Three regimes:

``EXACT``
    A deterministic measurement whose published digits the shipped command
    reproduces. Compared to the precision the paper prints.

A float
    A deterministic measurement compared within an absolute tolerance, because
    the last digit moves with the platform's float32 reduction order.

``SEED_SENSITIVE``
    A single draw of a statistic whose seed-to-seed spread exceeds the gap
    between the published digits. The paper prints it as if it were fixed. One
    such number exists and it is documented below rather than quietly widened.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

__all__ = [
    "EXACT",
    "SEED_SENSITIVE",
    "Expectation",
    "TABLE1_SYNTHETIC",
    "SEAM_LAG1",
    "SEC53_COST_DROP",
    "TABLE4_SPIRAL",
]

#: Compare to the precision the paper prints, with no slack.
EXACT: Final = 0.0

#: A number whose seed-to-seed spread is wider than its printed precision.
SEED_SENSITIVE: Final = None


@dataclass(frozen=True)
class Expectation:
    """One published number and how closely it has to be met.

    Attributes:
        value: What the paper prints.
        tolerance: Absolute slack. ``EXACT`` for none; ``SEED_SENSITIVE`` when
            the number is one draw of a noisy statistic, in which case
            ``spread`` says what the seed-to-seed range actually is.
        source: File and sentence in ``paper/ICLR Main/`` the value is quoted
            from.
        spread: Observed seed-to-seed range, for a ``SEED_SENSITIVE`` value.
        note: Anything a reader of a failing comparison needs to know.
    """

    value: float
    tolerance: float | None
    source: str
    spread: tuple[float, float] | None = None
    note: str = ""


# --- Table 1: analytical path geometry -------------------------------------
#
# The paper's table prints two columns per arm, both percentages: the share of
# paths whose peak angular velocity exceeds pi, and the same share weighted by
# each path's target energy. The quantiles the probe also prints are diagnostic
# and are not published, so they are not checked here.
TABLE1_SYNTHETIC: Final[dict[str, dict[str, Expectation]]] = {
    "Cartesian / independent": {
        "exceeds_pi": Expectation(
            0.460, EXACT, "tables/table1_bridges.tex, Synthetic / Cartesian, independent"
        ),
        "energy_share": Expectation(
            0.426, EXACT, "tables/table1_bridges.tex, Synthetic / Cartesian, independent"
        ),
    },
    "Cartesian / minibatch OT": {
        "exceeds_pi": Expectation(
            0.010, EXACT, "tables/table1_bridges.tex, Synthetic / Cartesian, minibatch OT"
        ),
        "energy_share": Expectation(
            0.001, EXACT, "tables/table1_bridges.tex, Synthetic / Cartesian, minibatch OT"
        ),
    },
    "Cylindrical / independent": {
        "exceeds_pi": Expectation(
            0.000, EXACT, "tables/table1_bridges.tex, Synthetic / Cylindrical (ours)"
        ),
        "energy_share": Expectation(
            0.000, EXACT, "tables/table1_bridges.tex, Synthetic / Cylindrical (ours)"
        ),
    },
}

# The tail index the introduction leans on: index one means P(peak > s) ~ c/s,
# for which the mean is infinite. Printed to three decimals by the probe; the
# paper states it qualitatively, so a loose band is the honest comparison.
TABLE1_TAIL_INDEX: Final = Expectation(
    1.0,
    0.05,
    "sections/05_experiments.tex -- the Cartesian tail has index one",
    note="An index near one is the claim; the third decimal is not.",
)


# --- Section 5.3: transport-cost saving against field size ------------------
#
# Quoted from the sentence "transport cost by 2.8% rather than 0.5%". Note that
# conf/experiment/sec53_ot_cost.yaml's expected: block still lists the pre-09-17
# figures (64x64 3.1%); the paper and the code agree, that block does not.
SEC53_COST_DROP: Final[dict[str, Expectation]] = {
    "64x64": Expectation(0.028, 0.002, "sections/05_experiments.tex -- 'transport cost by 2.8%'"),
    "320x320": Expectation(0.005, 0.002, "sections/05_experiments.tex -- 'rather than 0.5%'"),
}


# --- Table 4: the Factorised Coupling Trap ---------------------------------
#
# Mean over 8 seeds, spiral copula target, n = 1024. The caption says so, which
# is what makes these comparable at the precision printed.
TABLE4_SPIRAL: Final[dict[float, dict[str, Expectation]]] = {
    0.0: {
        "coupled_correlation": Expectation(
            0.0417, 0.002, "sections/05_experiments.tex Table 4, rho = 0.0"
        ),
        "cost_factorised": Expectation(0.1996, 0.002, "Table 4, rho = 0.0"),
        "cost_joint": Expectation(0.2055, 0.002, "Table 4, rho = 0.0"),
    },
    0.5: {
        "coupled_correlation": Expectation(0.0415, 0.002, "Table 4, rho = 0.5"),
        "cost_factorised": Expectation(0.1999, 0.002, "Table 4, rho = 0.5"),
        "cost_joint": Expectation(0.2248, 0.002, "Table 4, rho = 0.5"),
    },
    1.0: {
        "coupled_correlation": Expectation(0.0396, 0.002, "Table 4, rho = 1.0"),
        "cost_factorised": Expectation(0.2001, 0.002, "Table 4, rho = 1.0"),
        "cost_joint": Expectation(0.3845, 0.002, "Table 4, rho = 1.0"),
    },
}


# --- The one number that does not reproduce --------------------------------
#
# Section 5.4 states the lag-1 correlation across patch seams "falls from 0.979
# to 0.044". Measured here, at the seed the shipped command uses, it is -0.008.
# That is not a regression and not a bug: across eight seeds the statistic ranges
# over [-0.008, +0.055] with mean +0.017 and standard deviation 0.023, so 0.044
# is an ordinary draw and so is -0.008. The paper prints one draw of a noisy
# statistic in a sentence whose other two numbers (0.000 and 0.979) are stable to
# three decimals, which invites the reader to treat all three the same way.
#
# What the paper *claims* -- that seam correlation collapses from 0.979 to
# approximately zero -- holds under every seed. Only the digit is unreproducible.
# Checking it against 0.044 with a tight tolerance would fail honestly but
# uselessly; checking the collapse is the real test, and the published digit is
# recorded beside it so the discrepancy stays visible.
SEAM_LAG1: Final[dict[str, Expectation]] = {
    "published": Expectation(
        0.044,
        SEED_SENSITIVE,
        "sections/05_experiments.tex -- 'falls from 0.979 to 0.044'",
        spread=(-0.008, 0.055),
        note=(
            "One draw. Eight seeds give mean +0.017, sd 0.023; the shipped seed gives "
            "-0.008. The claim (collapse to ~0) reproduces; the digit does not."
        ),
    ),
    "inside_patches": Expectation(
        0.979, 0.002, "sections/05_experiments.tex -- 'correlation inside patches at 0.979'"
    ),
    "sliced_w2": Expectation(
        0.000, 0.001, "sections/05_experiments.tex -- 'pooled sliced W2 to the data at 0.000'"
    ),
    # What is actually being asserted, and what holds under every seed.
    "collapse_below": Expectation(
        0.10,
        EXACT,
        "sections/05_experiments.tex -- seam correlation collapses",
        note="Upper bound on |seam lag-1| after patch-level coupling, seed-independent.",
    ),
}

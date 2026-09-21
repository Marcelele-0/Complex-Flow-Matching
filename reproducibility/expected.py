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
    "TABLE5_ARMS",
    "TABLE5_DIFFUSION",
    "TABLE5_SLICED_W2",
    "TABLE5_SPATIAL_LAG1",
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


# --- Tables 2 and 3: the U-Net grids ---------------------------------------
#
# Sliced W2 against the data distribution, mean over 5 seeds, at each solver step
# count. Read off the tables themselves, which print three decimals; the
# tolerance is half of the last digit, so a mean that rounds to the printed value
# passes and one that does not, fails.
#
# The archives these are checked against are frozen input: tests/test_paper_results.py
# verifies every run's Hydra overrides against the experiment config it claims to
# come from, which is a different question from whether the numbers match the
# paper. This is that second question.
_UNET_TOLERANCE = 5e-4

#: Arm name in the archive -> the row of Table 2, at 64x64.
TABLE2_ARMS: Final[dict[str, str]] = {
    "un_euclidean_independent_scnull": "Cartesian, independent",
    "un_euclidean_ot_scnull": "Cartesian, minibatch OT",
    "un_cylindrical_independent_scnull": "Cylindrical, independent",
    "un_cylindrical_ot_scnull": "Cylindrical, minibatch OT",
}

#: Table 2, sliced W2 at 1, 2, 4, 8 and 100 steps.
TABLE2_SLICED_W2: Final[dict[str, dict[int, Expectation]]] = {
    arm: {
        steps: Expectation(
            value, _UNET_TOLERANCE, f"tables/table2_unet64.tex, {label}, {steps} step(s)"
        )
        for steps, value in zip((1, 2, 4, 8, 100), values, strict=True)
    }
    for arm, label, values in (
        (
            "un_euclidean_independent_scnull",
            "Cartesian / Independent",
            (0.416, 0.349, 0.285, 0.154, 0.046),
        ),
        ("un_euclidean_ot_scnull", "Cartesian / Minibatch OT", (0.376, 0.318, 0.260, 0.141, 0.047)),
        (
            "un_cylindrical_independent_scnull",
            "Cylindrical / Independent",
            (0.121, 0.148, 0.109, 0.066, 0.043),
        ),
        (
            "un_cylindrical_ot_scnull",
            "Cylindrical / Minibatch OT",
            (0.117, 0.114, 0.078, 0.054, 0.040),
        ),
    )
}

#: Table 3, the same metric across resolutions. The 64x64 block is Table 2's
#: rows, which is why that table's caption says it contains them.
TABLE3_SLICED_W2: Final[dict[str, dict[int, Expectation]]] = {
    arm: {
        steps: Expectation(
            value, _UNET_TOLERANCE, f"tables/table3_unet_scaling.tex, {label}, {steps} step(s)"
        )
        for steps, value in zip((1, 2, 4, 8, 100), values, strict=True)
    }
    for arm, label, values in (
        ("unsz_euclidean_ot_16", "16x16 Cartesian OT", (0.227, 0.202, 0.153, 0.097, 0.068)),
        (
            "unsz_cylindrical_independent_16",
            "16x16 CyFM independent",
            (0.116, 0.155, 0.133, 0.097, 0.062),
        ),
        ("unsz_cylindrical_ot_16", "16x16 CyFM joint OT", (0.098, 0.086, 0.053, 0.064, 0.054)),
        ("unsz_euclidean_ot_32", "32x32 Cartesian OT", (0.340, 0.275, 0.209, 0.104, 0.037)),
        (
            "unsz_cylindrical_independent_32",
            "32x32 CyFM independent",
            (0.123, 0.145, 0.113, 0.063, 0.047),
        ),
        ("unsz_cylindrical_ot_32", "32x32 CyFM joint OT", (0.113, 0.094, 0.072, 0.049, 0.041)),
    )
}

#: Seeds each U-Net table's caption says it averages over.
UNET_SEEDS: Final = 5


# --- Table 5: knee MRI at 64x64 --------------------------------------------
#
# Mean over 5 seeds on held-out test slices. The archive carries each arm twice,
# under a `dense_` and a `p5_` prefix; both give the same means, and `dense_` is
# the one used here.
#
# The score-based row needs its own tolerances: at k <= 2 its output is still the
# prior, whose scale is sigma_max = 53.4, so its "error" is tens rather than
# hundredths and a fixed absolute tolerance would be meaningless on both ends of
# the same column.
TABLE5_ARMS: Final[dict[str, str]] = {
    "p5_t5c64_cylindrical_ot": "Cylindrical + OT (ours)",
    "p5_t5c64_cylindrical_independent": "Cylindrical, independent",
    "p5_t5c64_euclidean_ot": "Cartesian + OT",
    "p5_t5c64_euclidean_independent": "Cartesian, independent",
}

TABLE5_SLICED_W2: Final[dict[str, dict[int, Expectation]]] = {
    arm: {
        steps: Expectation(
            value, 5e-5, f"tables/table5_fastmri64.tex, {TABLE5_ARMS[arm]}, k={steps}"
        )
        for steps, value in zip((1, 2, 4, 8, 100), values, strict=True)
    }
    for arm, values in (
        ("p5_t5c64_cylindrical_ot", (0.0593, 0.1200, 0.0937, 0.0735, 0.0722)),
        ("p5_t5c64_cylindrical_independent", (0.0604, 0.1338, 0.1064, 0.0756, 0.0677)),
        ("p5_t5c64_euclidean_ot", (0.1062, 0.1120, 0.0872, 0.0614, 0.0617)),
        ("p5_t5c64_euclidean_independent", (0.1518, 0.1432, 0.1054, 0.0582, 0.0563)),
    )
}

#: The VE-SDE baseline, whose column spans four orders of magnitude. Printed to
#: two decimals where it is large and four where it is small, so the tolerance
#: follows the printed precision rather than a single constant.
TABLE5_DIFFUSION: Final[dict[int, Expectation]] = {
    steps: Expectation(
        value,
        tolerance,
        f"tables/table5_fastmri64.tex, Complex diffusion (VE-SDE), k={steps}",
    )
    for steps, value, tolerance in (
        (1, 53.18, 5e-3),
        (2, 53.16, 5e-3),
        (4, 429.50, 5e-3),
        (8, 67.44, 5e-3),
        (100, 0.0771, 5e-5),
    )
}

#: Spatial lag-1 gap, the amplitude-texture row. This is the measure whose
#: ordering the paper reports as measured, including where the plane wins.
TABLE5_SPATIAL_LAG1: Final[dict[str, dict[int, Expectation]]] = {
    arm: {
        steps: Expectation(
            value, 5e-5, f"tables/table5_fastmri64.tex, spatial lag-1, {label}, k={steps}"
        )
        for steps, value in zip((1, 2, 4, 8, 100), values, strict=True)
    }
    for arm, label, values in (
        (
            "p5_t5c64_cylindrical_ot",
            "Cylindrical + OT (ours)",
            (0.2971, 0.1598, 0.0339, 0.0235, 0.0153),
        ),
        (
            "p5_t5c64_euclidean_independent",
            "Cartesian, independent",
            (0.0758, 0.0495, 0.0561, 0.0404, 0.0305),
        ),
    )
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

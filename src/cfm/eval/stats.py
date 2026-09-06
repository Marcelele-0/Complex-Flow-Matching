import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


@dataclasses.dataclass
class WilcoxonResult:
    metric: str
    baseline: str
    target: str
    n_pairs: int
    target_mean: float
    target_std: float
    baseline_mean: float
    baseline_std: float
    mean_diff: float
    median_diff: float
    statistic: float
    p_value: float
    p_value_adjusted: float
    effect_size_r: float
    significant_001: bool
    significant_05: bool


def load_eval_records(path: str | Path) -> pd.DataFrame:
    """Load evaluation records from a CSV file.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame containing evaluation records.

    Raises:
        ValueError: If a parquet file path is provided.
    """
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        raise ValueError(
            "Parquet format is not supported; eval_records are emitted as CSV. "
            "Please provide a CSV file."
        )
    return pd.read_csv(p)


def pair_evaluations(
    target_df: pd.DataFrame, baseline_df: pd.DataFrame, metric: str, on: str = "sample_id"
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pair evaluations for a specific metric.

    Args:
        target_df: Target evaluation records.
        baseline_df: Baseline evaluation records.
        metric: Metric to evaluate.
        on: Column to join on.

    Returns:
        Tuple of (target_values, baseline_values, sample_ids).

    Raises:
        ValueError: If fewer than 5 matching samples are found.
        pd.errors.MergeError: If duplicate sample keys violate one-to-one merge.
    """
    merged = pd.merge(
        target_df,
        baseline_df,
        on=on,
        suffixes=("_target", "_baseline"),
        how="inner",
        validate="one_to_one",
    )

    if len(merged) < 5:
        raise ValueError(f"Found {len(merged)} matching samples, need at least 5 for testing.")

    target_values = merged[f"{metric}_target"].to_numpy()
    baseline_values = merged[f"{metric}_baseline"].to_numpy()
    sample_ids = merged[on].tolist()

    return target_values, baseline_values, sample_ids


def compute_paired_wilcoxon(
    x: np.ndarray, y: np.ndarray, alternative: str = "two-sided"
) -> tuple[float, float, float]:
    """Compute Wilcoxon signed-rank test.

    Args:
        x: Target values.
        y: Baseline values.
        alternative: Test alternative hypothesis.

    Returns:
        Tuple of (statistic, p_value, effect_size_r).
        effect_size_r is signed; a positive value indicates target > baseline in median.
    """
    if len(x) == 0 or len(y) == 0:
        return 0.0, 1.0, 0.0

    if np.allclose(x, y, equal_nan=True):
        return 0.0, 1.0, 0.0

    diff = x - y
    n = len(diff)

    # Scipy 1.15.0+ wilcoxon
    try:
        res = wilcoxon(x, y, alternative=alternative)
        stat = float(res.statistic)
        p_value = float(res.pvalue)
    except ValueError:
        # Happens if all non-zero differences are zero or other degenerate cases
        return 0.0, 1.0, 0.0

    # Effect size r = Z / sqrt(N)
    # approximate Z from p-value or just use statistic
    from scipy.stats import norm

    # Clamp p_value to avoid exactly 0.0 which yields inf
    clamped_p = max(p_value, np.finfo(float).tiny)
    # Use ISF (Inverse Survival Function) instead of PPF (1 - p) to avoid float precision
    # making (1 - tiny) = 1.0 which results in inf Z-score.
    sign = float(np.sign(np.median(diff))) or 1.0
    z = norm.isf(clamped_p / 2) * sign
    effect_size_r = float(z / np.sqrt(n)) if n > 0 else 0.0

    return stat, p_value, effect_size_r


def apply_holm_bonferroni(results: list[WilcoxonResult]) -> list[WilcoxonResult]:
    """Apply step-down Holm-Bonferroni correction to a list of results.

    Args:
        results: List of Wilcoxon results.

    Returns:
        List of adjusted Wilcoxon results.
    """
    # Sort by p_value ascending
    sorted_idx = np.argsort([r.p_value for r in results])
    m = len(results)

    adjusted_p = np.zeros(m)
    for i, idx in enumerate(sorted_idx):
        adjusted_p[i] = results[idx].p_value * (m - i)
        if i > 0:
            adjusted_p[i] = max(adjusted_p[i], adjusted_p[i - 1])

    adjusted_p = np.minimum(adjusted_p, 1.0)

    for i, idx in enumerate(sorted_idx):
        results[idx].p_value_adjusted = float(adjusted_p[i])
        results[idx].significant_001 = results[idx].p_value_adjusted < 0.001
        results[idx].significant_05 = results[idx].p_value_adjusted < 0.05

    return results


def format_significance(p_val: float) -> str:
    """Format p-value significance as stars.

    Args:
        p_val: Adjusted p-value.

    Returns:
        String significance indicator.
    """
    if p_val < 0.001:
        return "***"
    if p_val < 0.01:
        return "**"
    if p_val < 0.05:
        return "*"
    return "ns"


_LATEX_ESCAPES = str.maketrans(
    {
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "$": r"\$",
    }
)


def escape_latex(text: str) -> str:
    """Escape characters that are special in LaTeX text mode."""
    return text.translate(_LATEX_ESCAPES)


def generate_latex_table(
    results: list[WilcoxonResult],
    target_name: str = "Cylindrical (Ours)",
    caption: str = "Statistical significance comparison",
    label: str = "tab:stats_significance",
) -> str:
    """Generate a LaTeX table from Wilcoxon results.

    Args:
        results: List of Wilcoxon results.
        target_name: Target model name.
        caption: Table caption.
        label: Table label.

    Returns:
        LaTeX table string.
    """
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        (
            f"Metric & Baseline & N & {escape_latex(target_name)} & Baseline & "
            "$\\Delta$ & $p$-value & Effect Size \\\\"
        ),
        "\\midrule",
    ]

    for r in results:
        sig = format_significance(r.p_value_adjusted)
        t_str = f"{r.target_mean:.3f} $\\pm$ {r.target_std:.3f}"
        b_str = f"{r.baseline_mean:.3f} $\\pm$ {r.baseline_std:.3f}"
        d_str = f"{r.mean_diff:.3f}"
        p_str = f"{r.p_value_adjusted:.1e}"
        e_str = f"{r.effect_size_r:.3f}"
        lines.append(
            f"{escape_latex(r.metric)} & {escape_latex(r.baseline)} & "
            f"{r.n_pairs} & {t_str} & {b_str} & "
            f"{d_str} & {p_str} ({sig}) & {e_str} \\\\"
        )

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table}",
        ]
    )

    return "\n".join(lines)

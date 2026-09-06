import argparse
import dataclasses
import json
import os
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


def process_stats(
    target_df: pd.DataFrame,
    baselines: dict[str, pd.DataFrame],
    metrics: list[str],
    target_name: str = "Target",
) -> list[WilcoxonResult]:
    """Compute Wilcoxon signed-rank tests across baselines and metrics.

    Args:
        target_df: Target evaluation records DataFrame.
        baselines: Dictionary mapping baseline names to evaluation DataFrames.
        metrics: List of metric column names to test.
        target_name: Name of target model.

    Returns:
        List of WilcoxonResult instances with adjusted p-values.

    Raises:
        KeyError: If metric is missing from target or any baseline.
    """
    results: list[WilcoxonResult] = []

    for metric in metrics:
        missing = [b_name for b_name, b_df in baselines.items() if metric not in b_df.columns]
        if metric not in target_df.columns or missing:
            common_sets = [set(target_df.columns)] + [set(b.columns) for b in baselines.values()]
            avail = (
                sorted(set.intersection(*common_sets)) if baselines else sorted(target_df.columns)
            )
            raise KeyError(
                f"Metric '{metric}' not found in evaluations. Available common metrics: {avail}"
            )

        for b_name, b_df in baselines.items():
            t_vals, b_vals, _ = pair_evaluations(target_df, b_df, metric)
            stat, p_val, eff = compute_paired_wilcoxon(t_vals, b_vals)

            res = WilcoxonResult(
                metric=metric,
                baseline=b_name,
                target=target_name,
                n_pairs=len(t_vals),
                target_mean=float(np.mean(t_vals)),
                target_std=float(np.std(t_vals)),
                baseline_mean=float(np.mean(b_vals)),
                baseline_std=float(np.std(b_vals)),
                mean_diff=float(np.mean(t_vals - b_vals)),
                median_diff=float(np.median(t_vals - b_vals)),
                statistic=stat,
                p_value=p_val,
                p_value_adjusted=p_val,
                effect_size_r=eff,
                significant_001=False,
                significant_05=False,
            )
            results.append(res)

    return apply_holm_bonferroni(results)


def print_table(results: list[WilcoxonResult]) -> None:
    """Print ASCII summary table of Wilcoxon significance test results.

    Args:
        results: List of WilcoxonResult instances.
    """
    print(
        f"{'Metric':<10} {'Baseline':<15} {'Target Mean':<15} "
        f"{'Base Mean':<15} {'Diff':<10} {'p-value':<12} {'Sig':<5}"
    )
    print("-" * 85)
    for r in results:
        sig = format_significance(r.p_value_adjusted)
        t_str = f"{r.target_mean:.3f}±{r.target_std:.3f}"
        b_str = f"{r.baseline_mean:.3f}±{r.baseline_std:.3f}"
        print(
            f"{r.metric:<10} {r.baseline:<15} {t_str:<15} {b_str:<15} "
            f"{r.mean_diff:>8.3f} {r.p_value_adjusted:>10.2e} {sig:<5}"
        )


def run_demo() -> None:
    """Run significance pipeline in demo mode with synthetic data."""
    print("Running in demo mode with synthetic data...")
    rng = np.random.default_rng(0)
    n_samples = 512
    sample_ids = [f"file_0[{i}]" for i in range(n_samples)]

    # Target data (slight advantage)
    target_psnr = rng.normal(35.5, 2.0, n_samples)
    baseline_psnr = target_psnr - rng.normal(0.5, 0.2, n_samples)

    target_df = pd.DataFrame({"sample_id": sample_ids, "psnr_db": target_psnr})
    baseline_df = pd.DataFrame({"sample_id": sample_ids, "psnr_db": baseline_psnr})

    metrics = ["psnr_db"]
    baselines = {"Baseline_Model": baseline_df}
    target_name = "Target_Model"

    results = process_stats(target_df, baselines, metrics, target_name)
    print_table(results)


def cli_main(argv: list[str] | None = None) -> int:
    """CLI entry point for automated statistical significance testing.

    Args:
        argv: Optional list of CLI arguments. Defaults to sys.argv[1:].

    Returns:
        Integer exit code (0 for success, non-zero for error).
    """
    parser = argparse.ArgumentParser(
        description="Automated statistical significance testing module."
    )
    parser.add_argument("--target", type=str, help="Path to target (ours) eval_records.csv")
    parser.add_argument(
        "--baselines", type=str, nargs="+", help="Baselines (name=path.csv or path.csv)"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["psnr_db", "ssim", "phase_error_rad"],
        help="Metrics to test",
    )
    parser.add_argument("--output-latex", type=str, help="Output file path for LaTeX table")
    parser.add_argument("--output-json", type=str, help="Output file path for JSON dump")
    parser.add_argument("--demo", action="store_true", help="Run with synthetic data")

    args = parser.parse_args(argv)

    if args.demo:
        run_demo()
        return 0

    if not args.target:
        print("Error: --target is required unless --demo is used.")
        return 1

    if not args.baselines:
        print("Error: --baselines is required unless --demo is used.")
        return 1

    target_df = load_eval_records(args.target)

    baselines = {}
    for b in args.baselines:
        if "=" in b:
            name, path = b.split("=", 1)
        else:
            path = b
            name = (
                os.path.basename(os.path.dirname(path))
                if os.path.basename(path) == "eval_records.csv"
                else os.path.basename(path)
            )
        baselines[name] = load_eval_records(path)

    results = process_stats(target_df, baselines, args.metrics)
    print_table(results)

    if args.output_latex:
        Path(args.output_latex).parent.mkdir(parents=True, exist_ok=True)
        latex_str = generate_latex_table(results)
        with open(args.output_latex, "w", encoding="utf-8") as f:
            f.write(latex_str)
        print(f"Wrote LaTeX table to {args.output_latex}")

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        out_dict = [dataclasses.asdict(r) for r in results]
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out_dict, f, indent=2)
        print(f"Wrote JSON stats to {args.output_json}")

    return 0

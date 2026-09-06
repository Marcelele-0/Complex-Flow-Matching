import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cfm.eval.stats import (
    WilcoxonResult,
    apply_holm_bonferroni,
    compute_paired_wilcoxon,
    format_significance,
    generate_latex_table,
    load_eval_records,
    pair_evaluations,
)


def run_demo() -> None:
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


def process_stats(
    target_df: pd.DataFrame,
    baselines: dict[str, pd.DataFrame],
    metrics: list[str],
    target_name: str = "Target",
) -> list[WilcoxonResult]:
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


def main() -> int:
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

    args = parser.parse_args()

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


if __name__ == "__main__":
    sys.exit(main())

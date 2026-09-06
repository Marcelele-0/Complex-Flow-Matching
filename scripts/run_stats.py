#!/usr/bin/env python3
"""Thin CLI wrapper for automated statistical significance testing."""

from __future__ import annotations

import sys

from cfm.eval.stats import (
    WilcoxonResult,
    apply_holm_bonferroni,
    cli_main,
    compute_paired_wilcoxon,
    format_significance,
    generate_latex_table,
    load_eval_records,
    pair_evaluations,
    print_table,
    process_stats,
    run_demo,
)

# Backwards compatibility alias
main = cli_main

__all__ = [
    "WilcoxonResult",
    "apply_holm_bonferroni",
    "cli_main",
    "compute_paired_wilcoxon",
    "format_significance",
    "generate_latex_table",
    "load_eval_records",
    "main",
    "pair_evaluations",
    "print_table",
    "process_stats",
    "run_demo",
]

if __name__ == "__main__":
    sys.exit(cli_main())

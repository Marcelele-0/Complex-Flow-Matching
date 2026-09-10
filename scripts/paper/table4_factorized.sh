#!/usr/bin/env bash
# Table 4 and Section 5.4 of the paper: the Factorized Coupling Trap. No network.
#
#   bash scripts/paper/table4_factorized.sh [OUTDIR]
#
# Part 1, Table 4 -- coupling amplitude and phase independently (scripts/coupling_gate.py,
# spiral copula target, n = 1024, 8 seeds). Expected, spiral rows:
#   rho 0.0  r_cl target 0.0242  coupled 0.0417  KS 0  cost factorised 0.1996  joint 0.2055
#   rho 0.5  r_cl target 0.3104  coupled 0.0415  KS 0  cost factorised 0.1999  joint 0.2248
#   rho 1.0  r_cl target 0.8670  coupled 0.0396  KS 0  cost factorised 0.2001  joint 0.3845
#
# Part 2, the patch seams -- section 5 of scripts/coupling_dimension.py (64x64 fields,
# correlation length 6, 16x16 patches). Expected:
#   patch-level OT  lag1 across seams 0.044  inside patches 0.979  sliced W2 0.0000
#   (data and image-level OT: 0.979 / 0.979 / 0.0000)
# The same run also prints the OT-cost-against-dimension numbers of Section 5.3; see
# scripts/paper/ot_cost_vs_dimension.sh.
#
# Every draw is seeded through an explicit torch.Generator.
set -eu
cd "$(dirname "$0")/../.."
OUT="${1:-outputs/paper}"
mkdir -p "$OUT"
uv run python scripts/coupling_gate.py --samples 1024 --seeds 8 --plot "$OUT/gate_a.png"
uv run python scripts/coupling_dimension.py

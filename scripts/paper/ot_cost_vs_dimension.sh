#!/usr/bin/env bash
# Section 5.3 and the third contribution of the paper: how much transport cost
# minibatch OT saves over independent pairing, against field size. No network.
#
#   bash scripts/paper/ot_cost_vs_dimension.sh
#
# Cylindrical metric, pointwise copula target (rho 0.5), white prior, batch 64.
# Expected, section 1 of the output ("cost drop"):
#   1x1 85.9%   2x2 72.1%   4x4 43.7%   8x8 21.3%   16x16 11.5%   32x32 5.1%   64x64 3.1%
# and in every row the assignment still reorders ~95-100% of the batch.
# Later sections (batch size, smooth data, smooth prior, patch seams) are recorded in
# docs/COUPLING_NOTES.md section 6; the patch seams are Table 4's second part.
set -eu
cd "$(dirname "$0")/../.."
uv run python scripts/coupling_dimension.py "$@"

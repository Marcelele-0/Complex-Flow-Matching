#!/usr/bin/env bash
# Table 1 of the paper: peak angular velocity of the analytic bridges. No network.
#
#   bash scripts/paper/table1_bridges.sh
#
# 10^6 endpoint pairs, copula target rho = 0.5, minibatch OT on scalar pairs with
# batch 256, seed 0. Exact closed form for the Cartesian chord, checked against a
# 20001-point grid. Deterministic: a rerun prints the same numbers.
#
# Expected (section 1 and 3 of the output):
#   Cartesian / independent   median 2.814   q99.9 1565.655   > pi 46.0%
#   Cartesian / minibatch OT  median 0.314   q99.9   26.716   > pi  1.0%
#   Cylindrical / independent median 1.570   q99.9    3.138   > pi  0.0%
#   tail index (Hill, top 1%): Cartesian / independent 1.006
set -eu
cd "$(dirname "$0")/../.."
uv run python scripts/bridge_angular_velocity.py "$@"

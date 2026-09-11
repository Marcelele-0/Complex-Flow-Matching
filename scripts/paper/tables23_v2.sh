#!/usr/bin/env bash
# Tables 2 and 3 of the revised paper, and its appendix table, retrained on one GPU.
#
#   bash scripts/paper/tables23_v2.sh                   # matched loss: Tables 2 and 3
#   LOSS_MODE=l1u bash scripts/paper/tables23_v2.sh     # appendix: cylinder with L1
#   LOSS_MODE=l1w bash scripts/paper/tables23_v2.sh     # appendix: Cartesian with L1
#
# Trains and evaluates every run of the grid {64, 32, 16} x geometries x
# {independent, joint OT} x seeds, one after another, through run_arm.sh; the
# protocol is conf/experiment/paper_unet*.yaml. A run whose evaluation already
# exists is skipped, so an interrupted sweep resumes where it stopped.
#
# Runtime on an RTX 4070 Ti SUPER: ~20 min per 64x64 run, ~3 min per 32x32 or 16x16
# run, so ~1.7 h per seed for the matched loss (both geometries) and ~9 h for all
# five seeds; each L1 variant covers one geometry and takes about half of that.
#
# Environment overrides:
#   LOSS_MODE  l2u (default) | l1u | l1w -- see run_arm.sh
#   GEOMS      default: both for l2u, cylindrical for l1u, euclidean for l1w
#   SEEDS      "0 1 2 3 4"     SIDES  "64 32 16"     COUPLINGS  "independent ot"
#   EPOCHS     40 (lower only for a smoke test)
#   PREFIX     run-name prefix, default "<LOSS_MODE>_"; use e.g. smoke_ for short runs
#
# Afterwards, freeze the evaluations and print or typeset the tables:
#   uv run python scripts/paper/paper_tables.py --export --prefix l2u_ --seeds 0 1 2 3 4 \
#       --archive-file docs/reproduce/paper_results/unet_eval_metrics_l2u.json
#   uv run python scripts/paper/loss_protocols.py --archive
#   uv run python scripts/paper/latex_tables.py
set -u
cd "$(dirname "$0")/../.."

export LOSS_MODE="${LOSS_MODE:-l2u}"
case "$LOSS_MODE" in
  l2u) default_geoms="cylindrical euclidean" ;;
  l1u) default_geoms="cylindrical" ;;
  l1w) default_geoms="euclidean" ;;
  *) echo "unknown LOSS_MODE=$LOSS_MODE (expected l2u, l1u or l1w)"; exit 2 ;;
esac
GEOMS="${GEOMS:-$default_geoms}"
SEEDS="${SEEDS:-0 1 2 3 4}"
SIDES="${SIDES:-64 32 16}"
COUPLINGS="${COUPLINGS:-independent ot}"
EPOCHS="${EPOCHS:-40}"
PREFIX="${PREFIX:-${LOSS_MODE}_}"

for SIDE in $SIDES; do
  for M in $GEOMS; do
    for C in $COUPLINGS; do
      for SEED in $SEEDS; do
        if [ "$SIDE" = 64 ]; then
          NAME="${PREFIX}un_${M}_${C}_scnull_s${SEED}"
        else
          NAME="${PREFIX}unsz_${M}_${C}_${SIDE}_s${SEED}"
        fi
        if compgen -G "outputs/evaluate/${NAME}_eval/*/metrics.json" > /dev/null; then
          echo "[$(date '+%H:%M:%S')] skip $NAME (evaluation exists)"
          continue
        fi
        bash scripts/paper/run_arm.sh "$NAME" "$SIDE" "$M" "$C" "$SEED" "$EPOCHS"
        code=$?
        if [ "$code" -ne 0 ]; then echo "FAILED: $NAME (exit $code)"; exit "$code"; fi
      done
    done
  done
done
echo "[$(date '+%H:%M:%S')] done"

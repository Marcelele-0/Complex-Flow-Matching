#!/bin/bash
# Trains and scores both geometries under identical settings, for Table 1.
#
# The two runs share everything except `manifold=`: same slices, same U-Net, same
# optimizer, same schedule, same seed, same metrics. That is the whole point.
#
# Usage:
#   ./schedule_comparison.sh                 # default tag, 100 epochs
#   TAG=ablation_mse EXTRA="training.loss.vel_loss_type=mse" ./schedule_comparison.sh
set -e

TAG="${TAG:-table1}"
T_START="${T_START:-0.5}"
EXTRA="${EXTRA:-}"

echo "Comparison run '${TAG}'"
echo "Start: $(date)"

# --- Training -------------------------------------------------------------
# Sequential by default: on one GPU these would contend for memory anyway. The
# experiment_name interpolates the manifold, so each arm gets its own output
# directory and its own W&B run without a second variable to keep in sync.
for MANIFOLD in cylindrical euclidean; do
  echo ""
  echo "=== training: ${MANIFOLD} ==="
  uv run src/cfm/train.py \
    manifold="${MANIFOLD}" \
    logging.experiment_name="${TAG}_${MANIFOLD}" \
    ${EXTRA}
done

# Hydra can do both arms in one command instead, if you prefer a sweep dir:
#   uv run src/cfm/train.py -m manifold=cylindrical,euclidean \
#     'logging.experiment_name=${TAG}_${manifold.name}'

# --- Evaluation -----------------------------------------------------------
# Same t_start, same split and same seed for both, so each arm is scored on the
# same slices, in the same order, with the same integration schedule.
#
# Both arms draw x_0 from the same distribution - the Euclidean prior defaults to
# 'uniform', which is the cylindrical prior's law expressed in (Re, Im) - so the
# geometry is the only variable in the table.
#
# They are distribution-matched, not sample-matched: the two draw the same law,
# not the same field. For the sample-paired version run the Euclidean arm on its
# own with `manifold.noise_prior=matched`. It cannot go in EXTRA: the key does not
# exist on conf/manifold/cylindrical.yaml, so Hydra rejects it for that arm.
for MANIFOLD in cylindrical euclidean; do
  echo ""
  echo "=== evaluating: ${MANIFOLD} ==="
  uv run src/cfm/evaluate.py \
    manifold="${MANIFOLD}" \
    evaluate.run_name="${TAG}_${MANIFOLD}" \
    evaluate.t_start="${T_START}" \
    logging.experiment_name="${TAG}_${MANIFOLD}"
done

echo ""
echo "Done: $(date)"
echo "Metrics: outputs/evaluate/${TAG}_{cylindrical,euclidean}/*/metrics.json"

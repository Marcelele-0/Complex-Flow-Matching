#!/bin/bash
set -e

echo "Starting experiments..."
echo "Start: $(date)"

uv run src/cfm/train.py loss.lambda_phase=2.0 logging.experiment_name=logit_normal_lp_2.0
uv run src/cfm/train.py loss.lambda_phase=3.0 logging.experiment_name=logit_normal_lp_3.0

echo "Done: $(date)"

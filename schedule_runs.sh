#!/bin/bash
set -e

echo "Starting CFM High-Frequency Boost experiments..."
echo "Start: $(date)"

# Baseline: No high-frequency boosting
uv run src/cfm/train.py logging.experiment_name=exp_001_baseline

# HF Boost v1: Lambda 0.5, standard boost factor
uv run src/cfm/train.py training.loss.lambda_hf=0.5 logging.experiment_name=exp_002_hf_boost_0.5

# HF Boost v2: Lambda 1.0, stronger high-frequency emphasis
uv run src/cfm/train.py training.loss.lambda_hf=1.0 logging.experiment_name=exp_003_hf_boost_1.0

# HF Boost v3: Lambda 0.5 with stronger radial boost factor
uv run src/cfm/train.py training.loss.lambda_hf=0.5 training.loss.hf_boost_factor=6.0 logging.experiment_name=exp_004_hf_boost_0.5_factor_6.0

echo "Done: $(date)"

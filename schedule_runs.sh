#!/bin/bash
set -e

echo "Starting CFM High-Frequency Boost experiments..."
echo "Start: $(date)"

# Lambda values are scaled for the ortho-normalized FFT in the HF loss:
# norm="ortho" divides magnitudes by sqrt(H*W) (= 512 at 512x512), so the
# pre-ortho values 0.5 / 1.0 become 256 / 512. Run names carry an _ortho
# suffix so W&B does not mix them with pre-ortho runs (e.g. old exp_002).

# Baseline: No high-frequency boosting
uv run src/cfm/train.py logging.experiment_name=exp_001_baseline

# HF Boost v1: Lambda 256 (pre-ortho 0.5), standard boost factor
uv run src/cfm/train.py training.loss.lambda_hf=256 logging.experiment_name=exp_002_hf_boost_l256_ortho

# HF Boost v2: Lambda 512 (pre-ortho 1.0), stronger high-frequency emphasis
uv run src/cfm/train.py training.loss.lambda_hf=512 logging.experiment_name=exp_003_hf_boost_l512_ortho

# HF Boost v3: Lambda 256 with stronger radial boost factor
uv run src/cfm/train.py training.loss.lambda_hf=256 training.loss.hf_boost_factor=6.0 logging.experiment_name=exp_004_hf_boost_l256_f6_ortho

echo "Done: $(date)"

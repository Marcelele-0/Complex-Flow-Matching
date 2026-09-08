#!/usr/bin/env bash
# Quality-vs-cost frontier on the UNCONDITIONAL path.
#
# The conditional-bridge sweep answered a question whose answer was fixed by the
# setup: starting from a zero-filled image that already scores 22.83 dB, the model
# only has to apply a small correction, and a small correction is a one-step
# problem. Nothing is integrated, so nothing can be said about integration cost.
#
# Here the model starts from a half-noised target and has an actual trajectory to
# traverse, while the metrics stay paired and meaningful. This is the setting in
# which "a geodesic path is already straight, a Euclidean one is not" can show up
# at all -- and it is the generative track the project is moving to.
#
# Checkpoints are the unconditional mini_bench models from 2026-08-29, which
# predate the bridge key and therefore trained on the noise path.
set -u
cd /home/marcel/Programming/papers/cfm
LOG_DIR="$1"; mkdir -p "$LOG_DIR"
stamp() { date '+%H:%M:%S'; }

for M in cylindrical euclidean; do
  for N in 1 2 4 8 16 32 64 100; do
    echo "[$(stamp)] === $M · num_steps=$N ==="
    uv run python -m cfm.evaluate \
      dataset=skm_tea manifold="$M" \
      evaluate.t_start=0.5 \
      +evaluate.mask.acceleration=8 \
      evaluate.num_steps="$N" evaluate.max_samples=128 \
      evaluate.run_name="${M}_mini_bench" \
      logging.use_wandb=false logging.experiment_name="unfe_${M}_${N}" \
      > "$LOG_DIR/${M}_${N}.log" 2>&1
    echo "[$(stamp)] $M n=$N exit=$?"
  done
done
echo "[$(stamp)] sweep done"

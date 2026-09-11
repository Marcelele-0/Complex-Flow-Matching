#!/usr/bin/env bash
# Quality-vs-cost frontier on the CONDITIONAL bridge. Kept for the record.
#
# This sweep cannot answer the question it was written for. Starting from a
# zero-filled image that already scores 22.83 dB at R=8, the model applies a
# small correction, and a small correction is a one-step problem -- there is no
# trajectory to integrate. Measured: quality *decreases* monotonically with more
# solver steps (23.714 / 23.629 / 23.574 dB at 1 / 2 / 4 steps, cylindrical),
# because faithfully following a slightly wrong field drifts further than a
# single coarse Euler step does.
#
# Use scripts/nfe_sweep_unconditional.sh for the question about integration.
# See docs/notes/GEOMETRY_NOTES.md section 4.
set -u
cd /home/marcel/Programming/papers/cfm
LOG_DIR="$1"; mkdir -p "$LOG_DIR"
stamp() { date '+%H:%M:%S'; }

STEPS="1 2 4 8 16 32 64 100"

for M in cylindrical euclidean; do
  for N in $STEPS; do
    echo "[$(stamp)] === $M · num_steps=$N ==="
    uv run python -m cfm.evaluate \
      dataset=skm_tea manifold="$M" \
      training.bridge=aliased evaluate.t_start=0.0 \
      +evaluate.mask.acceleration=8 \
      evaluate.num_steps="$N" evaluate.max_samples=128 \
      evaluate.run_name="g63_${M}_R8" \
      logging.use_wandb=false logging.experiment_name="nfe_${M}_${N}" \
      > "$LOG_DIR/${M}_${N}.log" 2>&1
    echo "[$(stamp)] $M n=$N exit=$?"
  done
done
echo "[$(stamp)] sweep done"

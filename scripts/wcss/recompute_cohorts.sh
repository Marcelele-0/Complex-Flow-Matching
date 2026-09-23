#!/usr/bin/env bash
# Relaunch every cohort the paper reports, under a distinct prefix, to check that the
# refactor left the results where they were.
#
#   bash scripts/wcss/recompute_cohorts.sh
#
# Each cohort keeps its own launcher; this only fixes the four things a rerun has to
# get right and that are easy to get wrong from memory:
#
#   1. A PREFIX no archived run uses, so nothing overwrites the outputs the frozen
#      archives point at. `rr_` is that prefix.
#   2. Five seeds for the synthetic grid. Its launcher defaults to three, and the
#      published tables are over five.
#   3. The 320 knee grid stops at the flow arms (array 0-19). The launcher appends
#      five diffusion tasks after them, but the score-based baseline was only ever
#      run at 64, and Appendix D prints no diffusion row -- running it here would
#      start a new experiment rather than reproduce one.
#   4. The 64 knee grid needs SIGMA_MAX=53.44. The launcher defaults to 140.0, which
#      is the calibration for the 320 matrix; inheriting it would make that row
#      unfairly bad and the comparison meaningless.
#
# Speech needs EPOCHS=40 for the same reason: its launcher defaults to 2, and the
# published arm is t6e40.
set -euo pipefail
cd /path/to/project/CyFM

echo "synthetic, 3 sizes x 2 geometries x 2 couplings x 5 seeds = 60 tasks"
sbatch --array=0-59 --job-name=rr-l2u \
  --export=ALL,SEEDS=0\ 1\ 2\ 3\ 4,PREFIX=rr_l2u_ \
  scripts/wcss/paper_tables_l2u.sbatch

echo "knee 320, flow arms only = 20 tasks"
sbatch --array=0-19 --job-name=rr-t5 \
  --export=ALL,PREFIX=rr_t5_ \
  scripts/wcss/table5_fastmri.sbatch

echo "knee 64, flow arms and the score-based baseline = 25 tasks"
sbatch --array=0-24 --job-name=rr-t5c64 \
  --export=ALL,PREFIX=rr_t5c64_,KSPACE_CROP=64,SIGMA_MAX=53.44 \
  scripts/wcss/table5_fastmri.sbatch

echo "speech STFT, 40 epochs = 20 tasks"
sbatch --array=0-19 --job-name=rr-t6 \
  --export=ALL,PREFIX=rr_t6e40_,EPOCHS=40 \
  scripts/wcss/table6_audio.sbatch

#!/usr/bin/env bash
# One run of the U-Net experiments, train then evaluate, through conf/experiment/.
# scripts/paper/reproduce.py runs whole specs; this runs a single cell (the HPC cluster array uses it).
#
#   LOSS_MODE=l2u bash scripts/paper/run_arm.sh NAME SIDE MANIFOLD COUPLING SEED [EPOCHS]
#
# SIDE is the synthetic field size; pass `native` for a cohort whose size is fixed by
# the data (fastMRI), and no crop override is sent. Two environment variables cover
# the file-backed cohorts:
#   EXPERIMENT   a conf/experiment/ name, used instead of the LOSS_MODE mapping
#                (e.g. EXPERIMENT=table5_fastmri SIDE=native)
#   HOLDOUT_ROLE dataset.role for the evaluation only, so the reference fields come
#                from volumes no arm trained on (e.g. HOLDOUT_ROLE=holdout)
#   TRAIN_STORE  dataset.store for the training only, the other half of the split.
#   DATA_DIR     dataset.data_dir for both, so a job can stage the stores onto a
#                node-local disk and keep 40 epochs of random reads off lustre.
#   EVAL_STORE   dataset.store for the evaluation only. fastMRI ships its own
#                patient-disjoint train/val split, so Table 5 trains on train.h5 and
#                scores against val.h5 rather than hashing one store into two roles.
#                Without this the runner can only split a single store, and a grid
#                pointed at val.h5 would train on 814 slices instead of 5324 -- the
#                runs would finish and the numbers would look plausible.
#   BATCH        override the experiment's batch size. The protocol's value must be
#                the same for every arm of a table; this exists for a smaller GPU
#                and for smoke runs, and a table built with it is not comparable.
#   EXTRA_OVERRIDES
#                space-separated Hydra overrides appended to BOTH commands, for
#                settings an arm needs and the experiment config cannot carry
#                because they differ between arms of one table (the diffusion
#                baseline's manifold.nfe_mode and manifold.sigma_max). Applied to
#                train and evaluate alike, so the two cannot drift apart.
#
# LOSS_MODE:
#   l2u  (default) the fair loss: L2 in both geometries, unweighted phase term
#        (see conf/experiment/ablation_loss32.yaml)
#   l1w  the arXiv v1 loss: L1 in both geometries, amplitude-weighted phase term
#   l1u  L1 in both geometries, unweighted phase term -- the cylinder's
#        counterpart of Cartesian L1, which has no phase weighting to remove
#
# Runs from the repository root through uv; nothing cluster-specific is assumed.
set -u
cd "$(dirname "$0")/../.."
NAME="$1"
SIDE="$2"
M="$3"
C="$4"
SEED="$5"
EPOCHS="${6:-40}"
LOSS_MODE="${LOSS_MODE:-l2u}"
EXPERIMENT="${EXPERIMENT:-}"
HOLDOUT_ROLE="${HOLDOUT_ROLE:-}"
EVAL_STORE="${EVAL_STORE:-}"
TRAIN_STORE="${TRAIN_STORE:-}"
DATA_DIR="${DATA_DIR:-}"
# Extra Hydra overrides appended to both commands, space separated. Exists so an
# ablation can vary one setting without a second copy of this script drifting from
# the one the tables were produced with.
EXTRA="${EXTRA:-}"
# Evaluation step grid. The default is the one the paper tables report.
NFE="${NFE:-[1,2,4,8,100]}"
BATCH="${BATCH:-}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

# Everything but geometry, coupling, size and seed lives in conf/experiment/.
# The archived runs were launched with the equivalent explicit overrides;
# tests/test_paper_results.py checks that both compose to the same config.
if [ -z "$EXPERIMENT" ]; then
  case "$LOSS_MODE" in
    l2u) EXPERIMENT=paper_unet ;;
    l1u) EXPERIMENT=paper_unet_l1 ;;
    l1w) EXPERIMENT=paper_unet_v1loss ;;
    *) echo "unknown LOSS_MODE=$LOSS_MODE (expected l2u, l1u or l1w)"; exit 2 ;;
  esac
fi
COMMON=(+experiment="$EXPERIMENT" manifold="$M" training.coupling="$C")
if [ -n "$EXTRA" ]; then
  # Word splitting is wanted here: EXTRA carries several overrides.
  # shellcheck disable=SC2206
  COMMON+=($EXTRA)
fi
if [ "$SIDE" != native ]; then
  COMMON+=(dataset.crop_size="[$SIDE,$SIDE]")
fi
if [ -n "$BATCH" ]; then
  COMMON+=(training.batch_size="$BATCH")
fi
if [ -n "$DATA_DIR" ]; then
  COMMON+=(dataset.data_dir="$DATA_DIR")
fi
# Deliberately unquoted: the variable holds several overrides, and word splitting
# is how they become several arguments.
# shellcheck disable=SC2206
if [ -n "$EXTRA_OVERRIDES" ]; then
  COMMON+=($EXTRA_OVERRIDES)
fi
# dataset.store is set per command, never in COMMON: Hydra refuses the same override
# twice, so the evaluation cannot simply append its own on top of a shared one.
TRAIN_ONLY=()
if [ -n "$TRAIN_STORE" ]; then
  TRAIN_ONLY+=(dataset.store="$TRAIN_STORE")
fi
EVAL_ONLY=()
if [ -n "$HOLDOUT_ROLE" ]; then
  EVAL_ONLY+=(dataset.role="$HOLDOUT_ROLE")
fi
if [ -n "$EVAL_STORE" ]; then
  EVAL_ONLY+=(dataset.store="$EVAL_STORE")
fi

echo "[$(date '+%H:%M:%S')] train $NAME ($EXPERIMENT)"
uv run --no-sync python -m cyfm.train "${COMMON[@]}" ${TRAIN_ONLY[@]+"${TRAIN_ONLY[@]}"} \
  training.epochs="$EPOCHS" training.seed="$SEED" logging.experiment_name="$NAME"
code=$?
if [ "$code" -ne 0 ]; then echo "FAILED train $NAME exit=$code"; exit "$code"; fi

echo "[$(date '+%H:%M:%S')] evaluate $NAME"
uv run --no-sync python -m cyfm.evaluate "${COMMON[@]}" ${EVAL_ONLY[@]+"${EVAL_ONLY[@]}"} \
  evaluate.run_name="$NAME" evaluate.num_fields=64 evaluate.seed="$SEED" \
  evaluate.nfe="$NFE" logging.experiment_name="${NAME}_eval"
code=$?
if [ "$code" -ne 0 ]; then echo "FAILED evaluate $NAME exit=$code"; exit "$code"; fi
echo "[$(date '+%H:%M:%S')] done $NAME"

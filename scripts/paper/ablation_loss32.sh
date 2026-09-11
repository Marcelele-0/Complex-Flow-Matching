#!/usr/bin/env bash
# Loss ablation at 32x32: does the cylinder's phase deficit at convergence come
# from the loss rather than the geometry?
#
#   bash scripts/paper/ablation_loss32.sh LOGDIR
#
# Arms, all with joint OT coupling, the white prior and the Table 3 schedule:
#   cylindrical  {amplitude-weighted phase, unweighted} x {L1, L2}
#   euclidean    {L1, L2}
# x 3 seeds = 18 runs. 32x32 is where the Table 3 asymptote gap is largest
# (0.114/0.134 vs 0.046/0.055) with separated seeds; at 16x16 the seed spread is
# as large as the effect. About 2-3 minutes per run on an RTX 4070 Ti SUPER.
#
# The seed loop is outermost, so each completed seed is a full set of arms.
# Summarise with scripts/paper/ablation_tables.py.
#
# Environment overrides:
#   EPOCHS (40)  SIZE (8192)  FIELDS (64)  SEEDS ("0 1 2")  SIDE (32)  PREFIX ("")
set -u
cd "$(dirname "$0")/../.."
LOG="${1:?usage: $0 LOGDIR}"
mkdir -p "$LOG"
EPOCHS="${EPOCHS:-40}"
SIZE="${SIZE:-8192}"
FIELDS="${FIELDS:-64}"
SEEDS="${SEEDS:-0 1 2}"
SIDE="${SIDE:-32}"
PREFIX="${PREFIX:-}"

stamp() { date '+%H:%M:%S'; }
run() {
  local label="$1"; shift
  "$@" > "$LOG/${label}.log" 2>&1
  local code=$?
  echo "[$(stamp)] ${label} exit=${code}"
  if [ "$code" -ne 0 ]; then echo "FAILED: ${label}"; tail -5 "$LOG/${label}.log"; exit 1; fi
}

# One arm: NAME MANIFOLD LOSS-OVERRIDES...
arm() {
  local NAME="$1" M="$2"; shift 2
  echo "[$(stamp)] === $NAME ==="
  run "train_${NAME}" uv run python -m cfm.train \
    dataset=cylinder_toy_field model=c_unet manifold="$M" \
    training.bridge=noise training.coupling=ot \
    training.epochs="$EPOCHS" training.batch_size=64 \
    training.grad_clip=null training.compile=false training.seed="$SEED" \
    dataset.size="$SIZE" dataset.crop_size="[$SIDE,$SIDE]" dataset.coupling=0.5 \
    logging.use_wandb=false logging.experiment_name="$NAME" "$@"
  run "eval_${NAME}" uv run python -m cfm.evaluate \
    dataset=cylinder_toy_field model=c_unet manifold="$M" \
    training.bridge=noise training.coupling=ot \
    dataset.crop_size="[$SIDE,$SIDE]" dataset.coupling=0.5 \
    evaluate.run_name="$NAME" evaluate.num_fields="$FIELDS" evaluate.seed="$SEED" \
    evaluate.nfe='[1,2,4,8,100]' \
    logging.use_wandb=false logging.experiment_name="${NAME}_eval" "$@"
}

for SEED in $SEEDS; do
  for L in l1 l2; do
    for W in true false; do
      if [ "$W" = true ]; then TAG=weighted; else TAG=unweighted; fi
      arm "${PREFIX}abl_cylindrical_${L}_${TAG}_${SIDE}_s${SEED}" cylindrical \
        training.loss.amp_loss_type="$L" training.loss.phase_loss_type="$L" \
        training.loss.phase_amplitude_weighting="$W"
    done
    arm "${PREFIX}abl_euclidean_${L}_${SIDE}_s${SEED}" euclidean \
      training.loss.vel_loss_type="$L"
  done
done
echo "[$(stamp)] done"
uv run python scripts/paper/ablation_tables.py --prefix "$PREFIX" --side "$SIDE"

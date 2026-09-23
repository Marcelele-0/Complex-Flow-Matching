#!/usr/bin/env bash
# Turn LibriSpeech into the compact complex-STFT store Table 6 trains on, as one CPU job.
#
#   sbatch scripts/wcss/build_stft_store.sbatch
#   COHORTS="dev-clean" sbatch scripts/wcss/build_stft_store.sbatch   # 40 speakers, not 80
#
# The audio counterpart of build_knee_store.sbatch, and much smaller: the archives are
# public on openslr, so there are no signed links to expire, and there is no ESPIRiT.
# Download and conversion are minutes rather than hours, which is why this is one task
# rather than an array with a merge.
#
# Only about an hour of speech fills a training set of this size, so the binding
# constraint is the number of SPEAKERS, not the hours: the split is by speaker, and a
# cohort with few speakers cannot be split without leaking a voice across it. dev-clean
# and test-clean give 40 each and their speaker ids are disjoint.
#SBATCH --job-name=cyfm-stft-store
#SBATCH --partition=bem2-cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/project/CyFM/logs/%x-%j.txt
#SBATCH --error=/path/to/project/CyFM/logs/%x-%j.err
set -u
# The default is the cluster copy; CYFM_ENV lets the test suite point this at a stub
# so the script's logic can be exercised without a cluster and without downloading.
source "${CYFM_ENV:-/path/to/project/CyFM/scripts/wcss/env.sh}"

COHORTS="${COHORTS:-dev-clean test-clean}"
OUT="${OUT:-$CYFM_ROOT/data/librispeech_stft/librispeech.h5}"
SPLIT="${SPLIT:-clean}"
BASE_URL="${BASE_URL:-https://www.openslr.org/resources/12}"

# $TMPDIR on the CPU partition is lustre, not a node-local disk: these nodes report
# TmpDisk=0. That is fine here -- the corpus is under a gigabyte -- but it is why this
# does not claim to stage anywhere faster.
WORK="${TMPDIR:-/tmp/$USER}/stft-$SLURM_JOB_ID"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/raw" "$(dirname "$OUT")"

echo "job=$SLURM_JOB_ID host=$(hostname) cohorts=$COHORTS out=$OUT"
df -h "$WORK" | tail -1

t0=$(date +%s)
for cohort in $COHORTS; do
  url="$BASE_URL/$cohort.tar.gz"
  status="$(curl -s -o /dev/null -w '%{http_code}' -r 0-0 --max-time 60 "$url")"
  case "$status" in
    200|206) ;;
    *) echo "$cohort: $url answered $status, not downloading"; exit 1 ;;
  esac
  wget -q -c "$url" -O "$WORK/$cohort.tar.gz" || { echo "$cohort: download failed"; exit 1; }
  echo "downloaded $cohort seconds=$(( $(date +%s) - t0 )) size=$(du -h "$WORK/$cohort.tar.gz" | cut -f1)"
  tar -xzf "$WORK/$cohort.tar.gz" -C "$WORK" || { echo "$cohort: extract failed"; exit 1; }
  rm -f "$WORK/$cohort.tar.gz"

  # LibriSpeech unpacks as LibriSpeech/<cohort>/<speaker>/<chapter>/*.flac, and the
  # builder reads the speaker from the first path component below --raw-dir. Pointing it
  # at the archive root would therefore see one "speaker" per cohort. Speaker ids are
  # globally unique across these cohorts, so moving each speaker directory up into one
  # flat root gives the builder what it expects and pools both cohorts into one split.
  src="$WORK/LibriSpeech/$cohort"
  [ -d "$src" ] || { echo "$cohort: expected $src after extraction"; exit 1; }
  for speaker in "$src"/*; do
    [ -d "$speaker" ] && mv "$speaker" "$WORK/raw/"
  done
  rm -rf "$WORK/LibriSpeech/$cohort"
done

echo "speakers=$(find "$WORK/raw" -maxdepth 1 -mindepth 1 -type d | wc -l) utterances=$(find "$WORK/raw" -name '*.flac' | wc -l)"

t1=$(date +%s)
uv run --no-sync python "$CYFM_ROOT/scripts/data/build_stft_store.py" \
  --raw-dir "$WORK/raw" \
  --out "$OUT" \
  --split "$SPLIT" \
  --delete-consumed
code=$?
echo "build_exit=$code seconds=$(( $(date +%s) - t1 )) total_seconds=$(( $(date +%s) - t0 ))"
exit "$code"

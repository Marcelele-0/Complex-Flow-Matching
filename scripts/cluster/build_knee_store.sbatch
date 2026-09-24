#!/usr/bin/env bash
# Turn one raw fastMRI archive into one knee CORPD part-store, as a CPU array job.
#
#   sbatch scripts/cluster/build_knee_store.sbatch                 # archive 0 only
#   sbatch --array=0-4 scripts/cluster/build_knee_store.sbatch     # archives 0..4
#
# Downloads, extraction and ESPIRiT are CPU work and must not sit on an H100: the GPU
# grant is the scarce resource (~7.5k GPU-h left against ~45k CPU-h), so this runs on
# cpu-normal. Everything heavy stays on node-local scratch and only the part-store
# (a few hundred MB) is written to lustre.
#
# One task per archive because the archives are independent and each is ~90 GB; tasks
# cannot share an HDF5 file, so each writes its own part and a single merge joins them
# once the array has drained:
#
#   uv run python scripts/data/build_knee_pd_store.py \
#       --merge $CYFM_ROOT/data/knee_pd/parts/train/*.h5 \
#       --out $CYFM_ROOT/data/knee_pd/train.h5 --split train
#
# The split is read from the archive name (knee_multicoil_train_batch_N -> train,
# knee_multicoil_val -> val) and every split gets its own parts directory, so all of
# FASTMRI_FULL_URLS can run as one array. The val archive's task rebuilds the validation
# store; merge parts/val/*.h5 into val.h5 with --split val. SPLIT=... overrides this.
#
# Archive URLs come from FASTMRI_FULL_URLS in $CYFM_ROOT/.env, which is where issue #77
# puts them and the only place the team maintains them. They are parsed with the same
# python-dotenv call and the same comma-or-newline splitting cyfm.data.download uses, so
# there is one format to get right rather than two. Task N takes the Nth URL.
#
#   URLS_VAR=FASTMRI_MINI_URLS sbatch scripts/cluster/build_knee_store.sbatch
#
# NYU signs these for about two weeks, so an expired link is the normal failure. The job
# checks its URL's signature (a 1-byte ranged GET -- a real HEAD fails 403 on these
# regardless of validity, since NYU's presigned links are signed for GET specifically)
# before pulling ~90 GB, and stops if that check fails.
#
# RUN ONE ARCHIVE FIRST. The log prints espirit_seconds_per_volume, which is what sizes
# --time for the rest, and node-local scratch on this partition is unverified -- the
# 7 TB figure in the HPC cluster skill is documented for GPU nodes.
#SBATCH --job-name=cyfm-knee-store
#SBATCH --partition=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00:00
#SBATCH --array=0-0
#SBATCH --output=/path/to/project/CyFM/logs/%x-%A_%a.txt
#SBATCH --error=/path/to/project/CyFM/logs/%x-%A_%a.err
set -u
source /path/to/project/CyFM/scripts/cluster/env.sh

ENV_FILE="${ENV_FILE:-$CYFM_ROOT/.env}"
URLS_VAR="${URLS_VAR:-FASTMRI_FULL_URLS}"
PARTS_ROOT="${PARTS_ROOT:-$CYFM_ROOT/data/knee_pd/parts}"

if [ ! -f "$ENV_FILE" ]; then
  echo "no $ENV_FILE; copy .env.example to .env and put the signed links in $URLS_VAR (#77)"
  exit 2
fi

# A read loop rather than mapfile: this stays valid on bash 3.2, so the script can be
# exercised off the cluster.
URLS=()
while IFS= read -r line; do
  URLS+=("$line")
done < <(uv run --no-sync python "$CYFM_ROOT/scripts/data/fastmri_urls.py" \
           --env "$ENV_FILE" --var "$URLS_VAR")

if [ "${#URLS[@]}" -eq 0 ]; then
  echo "$URLS_VAR in $ENV_FILE holds no URLs; the links expire in ~2 weeks (#77)"
  exit 2
fi
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#URLS[@]}" ]; then
  echo "task $SLURM_ARRAY_TASK_ID out of range: $URLS_VAR lists ${#URLS[@]} archive(s)"
  exit 2
fi
URL="${URLS[$SLURM_ARRAY_TASK_ID]}"

# The archive name identifies the part, so a rerun overwrites its own output rather
# than adding a duplicate the merge would then refuse.
ARCHIVE="$(basename "${URL%%\?*}")"
PART="${ARCHIVE%%.tar*}"

# FASTMRI_FULL_URLS lists the val archive beside the train batches, so the split comes
# from the archive rather than from a default: a val part stamped "train" would be
# merged into the training store and then scored as held out. Each split gets its own
# parts directory, so a `parts/train/*.h5` glob cannot pick a val part up. The test
# archives are refused: they ship without fully sampled k-space.
if [ -z "${SPLIT:-}" ]; then
  case "$PART" in
    *_train*) SPLIT=train ;;
    *_val*) SPLIT=val ;;
    *)
      echo "cannot tell the split of $ARCHIVE; set SPLIT=train or SPLIT=val explicitly"
      exit 2
      ;;
  esac
fi
PARTS_DIR="$PARTS_ROOT/$SPLIT"

# Node-local, so extraction and the ESPIRiT sidecars never touch lustre. Slurm cleans
# $TMPDIR up itself; the trap covers the case where it is not set.
WORK="${TMPDIR:-/tmp/$USER}/knee-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/extract" "$WORK/sens" "$PARTS_DIR"

echo "task=$SLURM_ARRAY_TASK_ID host=$(hostname) archive=$ARCHIVE part=$PART"
df -h "$WORK" | tail -1

# An expired signature is the normal failure here, and it costs nothing to find out
# before pulling ~90 GB rather than after. A real HEAD gets 403 on these regardless of
# validity -- NYU's presigned links are signed for GET -- so this checks with a 1-byte
# ranged GET instead, the same way scripts/data/fastmri_urls.py does.
STATUS="$(curl -s -o /dev/null -w '%{http_code}' -r 0-0 --max-time 60 "$URL")"
if [ "$STATUS" != "200" ] && [ "$STATUS" != "206" ]; then
  echo "GET $ARCHIVE -> $STATUS; this signed link is dead (#77). Check them all with:"
  echo "  uv run python scripts/data/fastmri_urls.py --check"
  exit 1
fi

t0=$(date +%s)
wget -q -c "$URL" -O "$WORK/$ARCHIVE" || { echo "download failed"; exit 1; }
echo "downloaded seconds=$(( $(date +%s) - t0 )) size=$(du -h "$WORK/$ARCHIVE" | cut -f1)"

t1=$(date +%s)
tar -xf "$WORK/$ARCHIVE" -C "$WORK/extract" || { echo "extract failed"; exit 1; }
rm -f "$WORK/$ARCHIVE"
echo "extracted seconds=$(( $(date +%s) - t1 ))"

# fastMRI archives unpack into a split directory (multicoil_train/, ...) but the builder
# globs one level, so point it at whichever directory actually holds the volumes.
FIRST_VOLUME="$(find "$WORK/extract" -name '*.h5' -print -quit)"
if [ -z "$FIRST_VOLUME" ]; then
  echo "no .h5 volumes inside $ARCHIVE"
  exit 1
fi
RAW_DIR="$(dirname "$FIRST_VOLUME")"
echo "raw_dir=$RAW_DIR volumes=$(find "$RAW_DIR" -name '*.h5' | wc -l)"

t2=$(date +%s)
uv run --no-sync python scripts/data/build_knee_pd_store.py \
  --raw-dir "$RAW_DIR" \
  --sens-dir "$WORK/sens" \
  --out "$PARTS_DIR/$PART.h5" \
  --split "$SPLIT" \
  --device cpu \
  --delete-rejected --delete-consumed --delete-consumed-maps
code=$?
echo "build_exit=$code seconds=$(( $(date +%s) - t2 )) total_seconds=$(( $(date +%s) - t0 ))"
exit "$code"

#!/bin/bash
# Synthesize a SUBSET of shards with Indic-Parler-TTS on the current GPU (for splitting
# a long Hindi audio run across parallel workers / multiple held GPUs). Resume-safe.
# Usage (via srun --overlap into a held job):
#   bash scripts/parler_range.sh <lang> <nshards> <shard_i> [<shard_i> ...]
#   e.g. A100: parler_range.sh hi 16 0 1 2 3 4 5 6 7
set -e
LANG_CODE="$1"; NSHARDS="$2"; shift 2; SHARDS="$@"
PROJ=/users/msingh/Girish/FraudAlignALM/multiling_fraud
OUTROOT="/users/msingh/sharedscratch/TeleAntiFraud_${LANG_CODE}"
DIAL="out/$LANG_CODE/dialogues_full.json"
echo "[parler-range] $(hostname) lang=$LANG_CODE shards=[$SHARDS]/$NSHARDS $(date)"
module load apps/anaconda3/2024.10/bin || true
. /opt/gridware/depots/54e7fb3c/el8/pkg/apps/anaconda3/2024.10/bin/etc/profile.d/conda.sh
export CONDA_ENVS_PATH=/mnt/scratch2/users/$USER/conda/envs
export HF_HOME=/mnt/scratch2/users/$USER/hf_cache HF_HUB_CACHE=/mnt/scratch2/users/$USER/hf_cache/hub
# indic-parler-tts is gated: read the HF token from an untracked file (never committed)
export HF_TOKEN="${HF_TOKEN:-$(cat /users/msingh/GPU/.hf_token 2>/dev/null)}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
cd "$PROJ"; conda activate "$CONDA_ENVS_PATH/fraudparler"
pids=(); k=0
for i in $SHARDS; do
  ( sleep $((k*8)); python scripts/tts_parler.py --lang "$LANG_CODE" --dialogues "$DIAL" \
      --audio-root "$OUTROOT" --shard "$i" --nshards "$NSHARDS" \
      --out "out/$LANG_CODE/withaudio_parler_${i}.json" ) &
  pids+=($!); k=$((k+1))
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "[parler-range] $(hostname) shards [$SHARDS] done (fail=$fail) $(date)"

#!/bin/bash
# Synthesize a SUBSET of shards on the current GPU (for splitting one dataset across
# multiple held GPUs). Usage (run via srun --overlap into each held job):
#   bash scripts/tts_range.sh <lang> <nshards> <shard_i> [<shard_i> ...]
#   e.g. GPU A: tts_range.sh en 12 0 1 2 3 4 5   |  GPU B: tts_range.sh en 12 6 7 8 9 10 11
set -e
LANG_CODE="$1"; NSHARDS="$2"; shift 2; SHARDS="$@"
PROJ=/users/msingh/Girish/FraudAlignALM/multiling_fraud
OUTROOT="/users/msingh/sharedscratch/TeleAntiFraud_${LANG_CODE}"
OUTDIR="out/$LANG_CODE"; DIAL="$OUTDIR/dialogues_full.json"
echo "[range] $(hostname) shards=[$SHARDS] of $NSHARDS $(date)"
module load apps/anaconda3/2024.10/bin || true
. /opt/gridware/depots/54e7fb3c/el8/pkg/apps/anaconda3/2024.10/bin/etc/profile.d/conda.sh
export CONDA_ENVS_PATH=/mnt/scratch2/users/$USER/conda/envs
export HF_HOME=/mnt/scratch2/users/$USER/hf_cache HF_HUB_CACHE=/mnt/scratch2/users/$USER/hf_cache/hub
export COQUI_TOS_AGREED=1
cd "$PROJ"; conda activate "$CONDA_ENVS_PATH/fraudtts"
pids=(); k=0
for i in $SHARDS; do
  ( sleep $((k*10)); python scripts/tts_synthesize.py --lang "$LANG_CODE" --dialogues "$DIAL" \
      --audio-root "$OUTROOT" --shard "$i" --nshards "$NSHARDS" --out "$OUTDIR/withaudio_${i}.json" ) &
  pids+=($!); k=$((k+1))
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "[range] $(hostname) shards [$SHARDS] done (fail=$fail) $(date)"

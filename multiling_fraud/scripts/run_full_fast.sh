#!/bin/bash
# FAST full run for a language, executed INTO an already-held GPU allocation:
#   srun --jobid=<held> --overlap bash scripts/run_full_fast.sh <lang> <N> <workers>
#   e.g.  scripts/run_full_fast.sh en 7177 4
# Generates N dialogues with the 72B (vLLM), runs NW parallel TTS workers on the single GPU
# (XTTS uses ~2GB of 80GB, so parallelism = speed), merges, assembles (fraud-only, 80/20).
set -e
LANG_CODE="${1:-en}"; N="${2:-7177}"; NW="${3:-4}"
PROJ=/users/msingh/Girish/FraudAlignALM/multiling_fraud
OUTROOT="/users/msingh/sharedscratch/TeleAntiFraud_${LANG_CODE}"
OUTDIR="out/$LANG_CODE"; DIAL="$OUTDIR/dialogues_full.json"
echo "[fast] $(hostname) lang=$LANG_CODE N=$N workers=$NW outroot=$OUTROOT $(date)"

module load apps/anaconda3/2024.10/bin || true
. /opt/gridware/depots/54e7fb3c/el8/pkg/apps/anaconda3/2024.10/bin/etc/profile.d/conda.sh
export CONDA_ENVS_PATH=/mnt/scratch2/users/$USER/conda/envs
export HF_HOME=/mnt/scratch2/users/$USER/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export COQUI_TOS_AGREED=1
GEN="$CONDA_ENVS_PATH/fraudgen"; TTS="$CONDA_ENVS_PATH/fraudtts"
MODEL="${FRAUD_LLM:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
cd "$PROJ"; mkdir -p "$OUTDIR"

# reuse dialogues if already generated (e.g. by gen_only.sh); otherwise generate now
if [ ! -s "$DIAL" ]; then
  echo "### Stage 1: generate $N dialogues ($LANG_CODE) with $MODEL"
  conda activate "$GEN"
  export VLLM_USE_FLASHINFER_SAMPLER=0
  [ -x "$GEN/lib/python3.10/site-packages/nvidia/cu13/bin/nvcc" ] && \
    { export CUDA_HOME="$GEN/lib/python3.10/site-packages/nvidia/cu13"; export PATH="$CUDA_HOME/bin:$PATH"; }
  python scripts/generate_dialogues.py --lang "$LANG_CODE" --n "$N" --backend vllm --model "$MODEL" --out "$DIAL"
  conda deactivate
else
  echo "### Stage 1: reusing existing $DIAL"
fi

echo "### Stage 2: $NW parallel TTS workers ($LANG_CODE)"
conda activate "$TTS"
pids=()
for i in $(seq 0 $((NW-1))); do
  python scripts/tts_synthesize.py --lang "$LANG_CODE" --dialogues "$DIAL" --audio-root "$OUTROOT" \
      --shard "$i" --nshards "$NW" --out "$OUTDIR/withaudio_${i}.json" &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" = 0 ] || { echo "[fast] a TTS worker failed"; exit 1; }

echo "### merge shards"
python - "$OUTDIR" <<'PY'
import json, glob, sys
outdir=sys.argv[1]; d=[]
for f in sorted(glob.glob(f"{outdir}/withaudio_*.json")):
    d.extend(json.load(open(f)))
json.dump(d, open(f"{outdir}/dialogues_full_withaudio.json","w"), ensure_ascii=False, indent=2)
print(f"[merge] {len(d)} dialogues with audio")
PY

echo "### Stage 3: assemble ($LANG_CODE, 80/20 split)"
python scripts/assemble_dataset.py --lang "$LANG_CODE" \
    --dialogues "$OUTDIR/dialogues_full_withaudio.json" --out-root "$OUTROOT" --test-frac 0.2
conda deactivate
echo "[fast] DONE $(date). dataset -> $OUTROOT"

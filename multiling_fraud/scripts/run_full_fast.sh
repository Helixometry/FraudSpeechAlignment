#!/bin/bash
# FAST full run, executed INTO an already-held GPU allocation via:
#   srun --jobid=<held> --overlap bash scripts/run_full_fast.sh 7177 full /users/msingh/sharedscratch/TeleAntiFraud_en 4
# Generates N dialogues with the 72B (vLLM), then runs NW parallel TTS workers on the
# single GPU (XTTS uses ~2GB of 80GB, so the card is under-used -> parallelism = speed),
# merges, and assembles into the original schema (fraud-only) with an 80/20 split.
set -e
N="${1:-7177}"; TAG="${2:-full}"; OUTROOT="${3:-/users/msingh/sharedscratch/TeleAntiFraud_en}"; NW="${4:-4}"
PROJ=/users/msingh/Girish/FraudAlignALM/multiling_fraud
echo "[fast] $(hostname) N=$N tag=$TAG outroot=$OUTROOT workers=$NW $(date)"

module load apps/anaconda3/2024.10/bin || true
. /opt/gridware/depots/54e7fb3c/el8/pkg/apps/anaconda3/2024.10/bin/etc/profile.d/conda.sh
export CONDA_ENVS_PATH=/mnt/scratch2/users/$USER/conda/envs
export HF_HOME=/mnt/scratch2/users/$USER/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export COQUI_TOS_AGREED=1
GEN="$CONDA_ENVS_PATH/fraudgen"; TTS="$CONDA_ENVS_PATH/fraudtts"
MODEL="${FRAUD_LLM:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
DIAL="out/dialogues_${TAG}.json"
cd "$PROJ"

echo "### Stage 1: generate $N dialogues with $MODEL (vLLM)"
conda activate "$GEN"
export VLLM_USE_FLASHINFER_SAMPLER=0
[ -x "$GEN/lib/python3.10/site-packages/nvidia/cu13/bin/nvcc" ] && \
  { export CUDA_HOME="$GEN/lib/python3.10/site-packages/nvidia/cu13"; export PATH="$CUDA_HOME/bin:$PATH"; }
python scripts/generate_dialogues.py --n "$N" --backend vllm --model "$MODEL" --out "$DIAL"
conda deactivate

echo "### Stage 2: $NW parallel TTS workers"
conda activate "$TTS"
pids=()
for i in $(seq 0 $((NW-1))); do
  python scripts/tts_synthesize.py --dialogues "$DIAL" --audio-root "$OUTROOT" \
      --shard "$i" --nshards "$NW" --out "out/withaudio_${TAG}_${i}.json" &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" = 0 ] || { echo "[fast] a TTS worker failed"; exit 1; }

echo "### merge shards -> withaudio"
python - "$TAG" <<'PY'
import json, glob, sys
tag=sys.argv[1]
d=[]
for f in sorted(glob.glob(f"out/withaudio_{tag}_*.json")):
    d.extend(json.load(open(f)))
json.dump(d, open(f"out/dialogues_{tag}_withaudio.json","w"), ensure_ascii=False, indent=2)
print(f"[merge] {len(d)} dialogues with audio")
PY

echo "### Stage 3: assemble (80/20 split)"
python scripts/assemble_dataset.py --dialogues "out/dialogues_${TAG}_withaudio.json" \
    --out-root "$OUTROOT" --test-frac 0.2
conda deactivate
echo "[fast] DONE $(date). dataset -> $OUTROOT"

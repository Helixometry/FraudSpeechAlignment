#!/bin/bash
# Generate ALL dialogues (text only) for a language — no TTS, no assemble.
# Run INTO the held GPU:  srun --jobid=<held> --overlap bash scripts/gen_only.sh <lang> <N>
#   e.g.  scripts/gen_only.sh en 7177     scripts/gen_only.sh hi 7177
# Output: out/<lang>/dialogues_full.json  (basis for BOTH audio and +/- preferences)
set -e
LANG_CODE="${1:-en}"; N="${2:-7177}"
PROJ=/users/msingh/Girish/FraudAlignALM/multiling_fraud
echo "[gen-only] $(hostname) lang=$LANG_CODE N=$N $(date)"
module load apps/anaconda3/2024.10/bin || true
. /opt/gridware/depots/54e7fb3c/el8/pkg/apps/anaconda3/2024.10/bin/etc/profile.d/conda.sh
export CONDA_ENVS_PATH=/mnt/scratch2/users/$USER/conda/envs
export HF_HOME=/mnt/scratch2/users/$USER/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
GEN="$CONDA_ENVS_PATH/fraudgen"
MODEL="${FRAUD_LLM:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
cd "$PROJ"; mkdir -p "out/$LANG_CODE"
conda activate "$GEN"
export VLLM_USE_FLASHINFER_SAMPLER=0
[ -x "$GEN/lib/python3.10/site-packages/nvidia/cu13/bin/nvcc" ] && \
  { export CUDA_HOME="$GEN/lib/python3.10/site-packages/nvidia/cu13"; export PATH="$CUDA_HOME/bin:$PATH"; }
python scripts/generate_dialogues.py --lang "$LANG_CODE" --n "$N" --backend vllm --model "$MODEL" \
    --out "out/$LANG_CODE/dialogues_full.json"
conda deactivate
echo "[gen-only] DONE $(date) -> $PROJ/out/$LANG_CODE/dialogues_full.json"

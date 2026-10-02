#!/bin/bash
# Full fresh dialogue generation for ONE language: over-generate with a buffer to
# absorb purity-filter drops, then trim to the EXACT per-type targets (7177 total).
#   srun --jobid=<held> --overlap bash scripts/full_gen.sh <lang> [buffer] [target]
set -e
LANG_CODE="${1:-en}"; BUFFER="${2:-8000}"; TARGET="${3:-7177}"
PROJ=/users/msingh/Girish/FraudAlignALM/multiling_fraud
echo "[full-gen] $(hostname) lang=$LANG_CODE buffer=$BUFFER target=$TARGET $(date)"
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
python scripts/generate_dialogues.py --lang "$LANG_CODE" --n "$BUFFER" --backend vllm --model "$MODEL" \
    --out "out/$LANG_CODE/dialogues_raw.json"
python scripts/trim_to_target.py --in "out/$LANG_CODE/dialogues_raw.json" \
    --out "out/$LANG_CODE/dialogues_full.json" --lang "$LANG_CODE" --total "$TARGET"
conda deactivate
echo "[full-gen] DONE $(date) -> $PROJ/out/$LANG_CODE/dialogues_full.json"

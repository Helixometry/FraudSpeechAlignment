#!/bin/bash
# Generate ALL dialogues (text only) with the 72B — no TTS, no assemble.
# Run INTO the held GPU: srun --jobid=<held> --overlap bash scripts/gen_only.sh 7177 full
# Output: out/dialogues_<tag>.json  (the basis for BOTH audio (later) and +/- preferences)
set -e
N="${1:-7177}"; TAG="${2:-full}"
PROJ=/users/msingh/Girish/FraudAlignALM/multiling_fraud
echo "[gen-only] $(hostname) N=$N tag=$TAG $(date)"
module load apps/anaconda3/2024.10/bin || true
. /opt/gridware/depots/54e7fb3c/el8/pkg/apps/anaconda3/2024.10/bin/etc/profile.d/conda.sh
export CONDA_ENVS_PATH=/mnt/scratch2/users/$USER/conda/envs
export HF_HOME=/mnt/scratch2/users/$USER/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
GEN="$CONDA_ENVS_PATH/fraudgen"
MODEL="${FRAUD_LLM:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
cd "$PROJ"
conda activate "$GEN"
export VLLM_USE_FLASHINFER_SAMPLER=0
[ -x "$GEN/lib/python3.10/site-packages/nvidia/cu13/bin/nvcc" ] && \
  { export CUDA_HOME="$GEN/lib/python3.10/site-packages/nvidia/cu13"; export PATH="$CUDA_HOME/bin:$PATH"; }
python scripts/generate_dialogues.py --n "$N" --backend vllm --model "$MODEL" --out "out/dialogues_${TAG}.json"
conda deactivate
echo "[gen-only] DONE $(date) -> $PROJ/out/dialogues_${TAG}.json"
python3 -c "import json;d=json.load(open('$PROJ/out/dialogues_${TAG}.json'));import collections;print('dialogues:',len(d),'| types:',dict(collections.Counter(x['fraud_type'] for x in d)))"

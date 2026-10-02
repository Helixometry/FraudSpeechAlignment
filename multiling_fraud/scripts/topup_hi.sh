#!/bin/bash
# Rebuild Hindi to 7177 fully-Devanagari (zero-Latin) dialogues:
#  - seed pool from the existing raw pool's strict-clean dialogues
#  - top-up ONLY the per-type deficit, one model-load per round, until targets met
#  - finalize: trim to exact per-type targets + renumber ids -> dialogues_full.json
#   srun --jobid=<A100 held> --overlap bash scripts/topup_hi.sh
set -e
PROJ=/users/msingh/Girish/FraudAlignALM/multiling_fraud
TOTAL=7177; MAXR=5
echo "[topup-hi] $(hostname) $(date)"
module load apps/anaconda3/2024.10/bin || true
. /opt/gridware/depots/54e7fb3c/el8/pkg/apps/anaconda3/2024.10/bin/etc/profile.d/conda.sh
export CONDA_ENVS_PATH=/mnt/scratch2/users/$USER/conda/envs
export HF_HOME=/mnt/scratch2/users/$USER/hf_cache; export HF_HUB_CACHE=$HF_HOME/hub
GEN="$CONDA_ENVS_PATH/fraudgen"; MODEL="${FRAUD_LLM:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
cd "$PROJ"; conda activate "$GEN"
export VLLM_USE_FLASHINFER_SAMPLER=0
[ -x "$GEN/lib/python3.10/site-packages/nvidia/cu13/bin/nvcc" ] && \
  { export CUDA_HOME="$GEN/lib/python3.10/site-packages/nvidia/cu13"; export PATH="$CUDA_HOME/bin:$PATH"; }

U=scripts/hi_topup_util.py
# back up the contaminated full set once
[ -f out/hi/dialogues_full.json ] && cp -n out/hi/dialogues_full.json out/hi/dialogues_full_contaminated.json
python "$U" init out/hi/dialogues_raw.json out/hi/pool.json

for r in $(seq 1 $MAXR); do
  python "$U" deficit out/hi/pool.json out/hi/counts.json $TOTAL
  if [ "$(cat out/hi/counts.json)" = "{}" ]; then echo "[topup-hi] no deficit — done"; break; fi
  echo "[topup-hi] === round $r ==="
  python scripts/generate_dialogues.py --lang hi --counts-json out/hi/counts.json \
      --backend vllm --model "$MODEL" --out "out/hi/round_$r.json"
  python "$U" merge out/hi/pool.json "out/hi/round_$r.json"
done

python "$U" finalize out/hi/pool.json out/hi/dialogues_full.json hi $TOTAL
conda deactivate
echo "[topup-hi] DONE $(date)"

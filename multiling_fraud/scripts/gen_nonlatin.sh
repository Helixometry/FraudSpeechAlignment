#!/bin/bash
# Fresh generation for a NON-LATIN language (ko/hi/…) to EXACTLY <total> fully-native
# (zero-Latin) dialogues. is_clean already rejects any Latin during generation, so:
#  1) generate a big initial buffer  -> strict-clean pool
#  2) top-up ONLY the per-type deficit, one model-load per round, until targets met
#  3) finalize: trim to per-type targets + renumber ids -> dialogues_full.json
#   srun --jobid=<held> --overlap bash scripts/gen_nonlatin.sh <lang> [initbuf] [total]
set -e
LANG_CODE="${1:-ko}"; INIT="${2:-9000}"; TOTAL="${3:-7177}"; MAXR=6
PROJ=/users/msingh/Girish/FraudAlignALM/multiling_fraud
echo "[gen-nl] $(hostname) lang=$LANG_CODE init=$INIT total=$TOTAL $(date)"
module load apps/anaconda3/2024.10/bin || true
. /opt/gridware/depots/54e7fb3c/el8/pkg/apps/anaconda3/2024.10/bin/etc/profile.d/conda.sh
export CONDA_ENVS_PATH=/mnt/scratch2/users/$USER/conda/envs
export HF_HOME=/mnt/scratch2/users/$USER/hf_cache; export HF_HUB_CACHE=$HF_HOME/hub
GEN="$CONDA_ENVS_PATH/fraudgen"; MODEL="${FRAUD_LLM:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
cd "$PROJ"; mkdir -p "out/$LANG_CODE"; conda activate "$GEN"
export VLLM_USE_FLASHINFER_SAMPLER=0
[ -x "$GEN/lib/python3.10/site-packages/nvidia/cu13/bin/nvcc" ] && \
  { export CUDA_HOME="$GEN/lib/python3.10/site-packages/nvidia/cu13"; export PATH="$CUDA_HOME/bin:$PATH"; }
U=scripts/hi_topup_util.py
O="out/$LANG_CODE"

# round 0: big initial buffer (is_clean drops any Latin -> raw is already clean)
python scripts/generate_dialogues.py --lang "$LANG_CODE" --n "$INIT" --backend vllm \
    --model "$MODEL" --out "$O/dialogues_raw.json"
python "$U" init "$O/dialogues_raw.json" "$O/pool.json"

for r in $(seq 1 $MAXR); do
  python "$U" deficit "$O/pool.json" "$O/counts.json" "$TOTAL"
  if [ "$(cat "$O/counts.json")" = "{}" ]; then echo "[gen-nl] no deficit — done"; break; fi
  echo "[gen-nl] === round $r ==="
  python scripts/generate_dialogues.py --lang "$LANG_CODE" --counts-json "$O/counts.json" \
      --backend vllm --model "$MODEL" --out "$O/round_$r.json"
  python "$U" merge "$O/pool.json" "$O/round_$r.json"
done

python "$U" finalize "$O/pool.json" "$O/dialogues_full.json" "$LANG_CODE" "$TOTAL"
rm -f "$O"/round_*.json "$O/pool.json" "$O/counts.json"
conda deactivate
echo "[gen-nl] DONE $(date) -> $PROJ/$O/dialogues_full.json"

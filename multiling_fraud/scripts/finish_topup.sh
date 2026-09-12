#!/bin/bash
# Top-up an already-generated dataset that fell short of per-type targets.
# Re-seeds the pool from the existing (clean, renumbered) dialogues_full.json,
# then runs generous deficit rounds until every type hits its target, and
# re-finalizes. Works for both non-Latin (ko/hi) and code-mix (hinglish) via
# CLEAN_MODE (exported below from the 2nd arg).
#   srun --jobid=<held> --overlap bash scripts/finish_topup.sh <lang> <mode> [total] [survival]
#   mode: nonlatin | codemix
set -e
LANG_CODE="${1:-ko}"; MODE="${2:-nonlatin}"; TOTAL="${3:-7177}"; SURV="${4:-0.5}"; MAXR=12
PROJ=/users/msingh/Girish/FraudAlignALM/multiling_fraud
echo "[topup-fin] $(hostname) lang=$LANG_CODE mode=$MODE total=$TOTAL surv=$SURV $(date)"
module load apps/anaconda3/2024.10/bin || true
. /opt/gridware/depots/54e7fb3c/el8/pkg/apps/anaconda3/2024.10/bin/etc/profile.d/conda.sh
export CONDA_ENVS_PATH=/mnt/scratch2/users/$USER/conda/envs
export HF_HOME=/mnt/scratch2/users/$USER/hf_cache; export HF_HUB_CACHE=$HF_HOME/hub
GEN="$CONDA_ENVS_PATH/fraudgen"; MODEL="${FRAUD_LLM:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
cd "$PROJ"; conda activate "$GEN"
export VLLM_USE_FLASHINFER_SAMPLER=0
export CLEAN_MODE="$MODE" TOPUP_SURVIVAL="$SURV"
[ -x "$GEN/lib/python3.10/site-packages/nvidia/cu13/bin/nvcc" ] && \
  { export CUDA_HOME="$GEN/lib/python3.10/site-packages/nvidia/cu13"; export PATH="$CUDA_HOME/bin:$PATH"; }
U=scripts/hi_topup_util.py
O="out/$LANG_CODE"

# seed pool from the existing finalized set (already clean + renumbered)
python "$U" init "$O/dialogues_full.json" "$O/pool.json"

for r in $(seq 1 $MAXR); do
  python "$U" deficit "$O/pool.json" "$O/counts.json" "$TOTAL"
  if [ "$(cat "$O/counts.json")" = "{}" ]; then echo "[topup-fin] no deficit — done"; break; fi
  echo "[topup-fin] === round $r ==="
  python scripts/generate_dialogues.py --lang "$LANG_CODE" --counts-json "$O/counts.json" \
      --backend vllm --model "$MODEL" --out "$O/tround_$r.json"
  python "$U" merge "$O/pool.json" "$O/tround_$r.json"
done

python "$U" finalize "$O/pool.json" "$O/dialogues_full.json" "$LANG_CODE" "$TOTAL"
rm -f "$O"/tround_*.json "$O/pool.json" "$O/counts.json"
conda deactivate
echo "[topup-fin] DONE $(date) -> $PROJ/$O/dialogues_full.json"

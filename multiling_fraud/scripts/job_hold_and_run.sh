#!/bin/bash
#SBATCH --job-name=fraud-a100-hold
#SBATCH --partition=k2-gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
#SBATCH --output=/users/msingh/GPU/%x-%j.out
#SBATCH --error=/users/msingh/GPU/%x-%j.out

# Persistent 3-day GPU hold. Two of these run (one A100, one H100); we keep BOTH.
# The FIRST to be granted grabs a lock and runs the priority 20-sample 72B validation.
# The other stays RESERVED (held, idle) for the next code. Neither is cancelled.
# GPU stays held even if a run fails. Run more work in without re-queue via:
#   srun --jobid=<this job> --overlap bash scripts/job_pilot_a100.sh <N> <tag> <outroot>

PROJ=/users/msingh/Girish/FraudAlignALM/multiling_fraud
LOCK=/users/msingh/GPU/fraud_validation.lock
echo "$(hostname) $SLURM_JOB_ID" > "/users/msingh/GPU/fraud_gpu_${SLURM_JOB_ID}.node"
echo "[hold] GPU granted on $(hostname), job $SLURM_JOB_ID, $(date)"
nvidia-smi -L || true

if mkdir "$LOCK" 2>/dev/null; then
  echo "$(hostname) $SLURM_JOB_ID" > "$LOCK/owner"
  echo "[hold] PRIMARY: running priority 20-sample 72B validation"
  bash "$PROJ/scripts/job_pilot_a100.sh" 20 test /users/msingh/sharedscratch/TeleAntiFraud_en_test \
    && echo "[hold] validation finished OK" \
    || echo "[hold] validation FAILED (GPU stays held for rerun)"
  echo "done" > "$LOCK/status"
else
  echo "[hold] SECONDARY: another GPU is primary; this card is RESERVED for further work"
fi

echo "[hold] ===================================================================="
echo "[hold] GPU HELD for 3 days (job $SLURM_JOB_ID on $(hostname)). Run work without re-queue:"
echo "[hold]   srun --jobid=$SLURM_JOB_ID --overlap bash $PROJ/scripts/job_pilot_a100.sh <N> <tag> <outroot>"
echo "[hold] ===================================================================="
sleep infinity

#!/bin/bash
#SBATCH --job-name=fraud-tts-asm
#SBATCH --partition=k2-gpu-a100mig
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=2:00:00
#SBATCH --output=/users/msingh/GPU/%x-%j.out
#SBATCH --error=/users/msingh/GPU/%x-%j.out

# Stages 2+3 only: TTS (XTTS-v2, proven) + assemble, on a pre-written dialogues file.
set -e
echo "[job] node=$(hostname) jobid=$SLURM_JOB_ID $(date)"
module load apps/anaconda3/2024.10/bin || true
. /opt/gridware/depots/54e7fb3c/el8/pkg/apps/anaconda3/2024.10/bin/etc/profile.d/conda.sh
export CONDA_ENVS_PATH=/mnt/scratch2/users/$USER/conda/envs
export HF_HOME=/mnt/scratch2/users/$USER/hf_cache
export COQUI_TOS_AGREED=1

PROJ=/users/msingh/Girish/FraudAlignALM/multiling_fraud
OUTROOT=/users/msingh/sharedscratch/TeleAntiFraud_en
DIAL="${1:-out/dialogues_tiny.json}"          # dialogues file (arg 1)
WITHAUDIO="${DIAL%.json}_withaudio.json"
cd "$PROJ"
conda activate "$CONDA_ENVS_PATH/fraudtts"

echo "### Stage 2: TTS synthesis (2-voice multi-turn) on $DIAL"
python scripts/tts_synthesize.py --dialogues "$DIAL" --audio-root "$OUTROOT" --out "$WITHAUDIO"

echo "### Stage 3: assemble dataset (original schema)"
python scripts/assemble_dataset.py --dialogues "$WITHAUDIO" --out-root "$OUTROOT"
conda deactivate
echo "[job] DONE $(date). audio -> $OUTROOT/audio/NEG-gen-en/ ; dataset -> $OUTROOT/{binary_classification,sft}/"

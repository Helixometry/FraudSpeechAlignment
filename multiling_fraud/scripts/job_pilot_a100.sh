#!/bin/bash
#SBATCH --job-name=fraud-pilot
#SBATCH --partition=k2-gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
#SBATCH --output=/users/msingh/GPU/%x-%j.out
#SBATCH --error=/users/msingh/GPU/%x-%j.out

# Full pipeline on a WHOLE A100 (no MIG -> vLLM works). Generate N scam dialogues
# with the 72B, synthesize 2-voice audio, assemble into the TeleAntiFraud schema.
#   sbatch job_pilot_a100.sh 20  test /users/msingh/sharedscratch/TeleAntiFraud_en_test   # 72B validation
#   sbatch job_pilot_a100.sh 7177 full /users/msingh/sharedscratch/TeleAntiFraud_en       # full run
# args: $1=N  $2=tag(default pilot)  $3=outroot(default .../TeleAntiFraud_en)
set -e
N="${1:-50}"
TAG="${2:-pilot}"
OUTROOT="${3:-/users/msingh/sharedscratch/TeleAntiFraud_en}"
echo "[job] node=$(hostname) jobid=$SLURM_JOB_ID N=$N tag=$TAG outroot=$OUTROOT $(date)"
nvidia-smi -L || true

module load apps/anaconda3/2024.10/bin || true
. /opt/gridware/depots/54e7fb3c/el8/pkg/apps/anaconda3/2024.10/bin/etc/profile.d/conda.sh
export CONDA_PKGS_DIRS=/mnt/scratch2/users/$USER/conda/pkgs
export CONDA_ENVS_PATH=/mnt/scratch2/users/$USER/conda/envs
export HF_HOME=/mnt/scratch2/users/$USER/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export COQUI_TOS_AGREED=1
mkdir -p "$CONDA_PKGS_DIRS" "$CONDA_ENVS_PATH" "$HF_HOME"

GEN="$CONDA_ENVS_PATH/fraudgen"; TTS="$CONDA_ENVS_PATH/fraudtts"
PROJ=/users/msingh/Girish/FraudAlignALM/multiling_fraud
MODEL="${FRAUD_LLM:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
DIAL="out/dialogues_${TAG}.json"
WITHAUDIO="out/dialogues_${TAG}_withaudio.json"
cd "$PROJ"

# --- ensure fraudgen (vLLM) ---
if [ ! -d "$GEN" ]; then
  conda create -y -p "$GEN" python=3.10
  conda activate "$GEN"; pip install --upgrade pip
  conda install -y -c conda-forge "rust>=1.85" maturin
  pip install "vllm>=0.6.3" "transformers>=4.45" autoawq accelerate
  conda deactivate
fi
# --- ensure fraudtts healthy ---
conda activate "$TTS"
python -c "import torch" 2>/dev/null || pip install torch torchaudio
python -c "from transformers.pytorch_utils import isin_mps_friendly" 2>/dev/null || pip install "transformers>=4.57,<5"
python -c "import torchcodec" 2>/dev/null || pip install torchcodec
conda deactivate

echo "### Stage 1: generate $N dialogues with $MODEL (vLLM, full A100)"
conda activate "$GEN"
# flashinfer JIT-compiles a CUDA sampler kernel needing nvcc (absent in base image).
# Use vLLM's native sampler (no compile) and expose the env's bundled nvcc as backup.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_FLASHINFER=0
if [ -x "$GEN/lib/python3.10/site-packages/nvidia/cu13/bin/nvcc" ]; then
  export CUDA_HOME="$GEN/lib/python3.10/site-packages/nvidia/cu13"
  export PATH="$CUDA_HOME/bin:$PATH"
fi
python scripts/generate_dialogues.py --n "$N" --backend vllm --model "$MODEL" --out "$DIAL"
conda deactivate

echo "### Stage 2: TTS synthesis (2-voice multi-turn)"
conda activate "$TTS"
python scripts/tts_synthesize.py --dialogues "$DIAL" --audio-root "$OUTROOT" --out "$WITHAUDIO"

echo "### Stage 3: assemble dataset"
python scripts/assemble_dataset.py --dialogues "$WITHAUDIO" --out-root "$OUTROOT"
conda deactivate
echo "[job] DONE $(date). dataset -> $OUTROOT ; dialogues -> $PROJ/$DIAL"

#!/bin/bash
# Build the two conda envs on scratch for the fraud-dataset pipeline.
# Run this ON THE A100 NODE after attaching:
#   srun --jobid=<JID> --overlap --pty bash
#   bash scripts/env_setup.sh
# vLLM and Coqui-TTS have conflicting deps, so we keep them separate.
set -euo pipefail

module load apps/anaconda3/2024.10/bin || true
. /opt/gridware/depots/54e7fb3c/el8/pkg/apps/anaconda3/2024.10/bin/etc/profile.d/conda.sh
export CONDA_PKGS_DIRS=/mnt/scratch2/users/$USER/conda/pkgs
export CONDA_ENVS_PATH=/mnt/scratch2/users/$USER/conda/envs
mkdir -p "$CONDA_PKGS_DIRS" "$CONDA_ENVS_PATH"

# HF cache on scratch (models are large)
export HF_HOME=/mnt/scratch2/users/$USER/hf_cache
mkdir -p "$HF_HOME"

echo "=== [1/2] fraudgen (vLLM + transformers) ==="
if ! conda env list | grep -q "/fraudgen"; then
  conda create -y -p "$CONDA_ENVS_PATH/fraudgen" python=3.10
fi
conda activate "$CONDA_ENVS_PATH/fraudgen"
pip install --upgrade pip
# vLLM pulls rust-built deps (llguidance) needing edition2024 -> need modern Rust.
conda install -y -c conda-forge "rust>=1.85" maturin
pip install "vllm>=0.6.3" "transformers>=4.45" autoawq accelerate
conda deactivate

echo "=== [2/2] fraudtts (Coqui XTTS-v2 + audio) ==="
if ! conda env list | grep -q "/fraudtts"; then
  conda create -y -p "$CONDA_ENVS_PATH/fraudtts" python=3.10
fi
conda activate "$CONDA_ENVS_PATH/fraudtts"
pip install --upgrade pip
# IMPORTANT: coqui-tts does NOT install torch itself -> install it explicitly first.
pip install torch torchaudio
# coqui-tts is the maintained fork of TTS. pydub needs ffmpeg.
# [codec] extra pulls torchcodec, required for audio IO on torch>=2.9.
pip install "coqui-tts[codec]" pydub soundfile
# coqui-tts 0.27 requires transformers>=4.57 but breaks on transformers 5.x
# (removed isin_mps_friendly). Pin to the 4.57 line which has the symbol.
pip install "transformers>=4.57,<5"
conda install -y -c conda-forge ffmpeg
conda deactivate

echo "=== DONE. Envs at $CONDA_ENVS_PATH (fraudgen, fraudtts). HF_HOME=$HF_HOME ==="

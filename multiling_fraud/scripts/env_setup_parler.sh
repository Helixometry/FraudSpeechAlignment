#!/bin/bash
# Build the fraudparler env (AI4Bharat Indic-Parler-TTS) on scratch.
set -e
module load apps/anaconda3/2024.10/bin || true
. /opt/gridware/depots/54e7fb3c/el8/pkg/apps/anaconda3/2024.10/bin/etc/profile.d/conda.sh
export CONDA_PKGS_DIRS=/mnt/scratch2/users/$USER/conda/pkgs
export CONDA_ENVS_PATH=/mnt/scratch2/users/$USER/conda/envs
ENV=$CONDA_ENVS_PATH/fraudparler
[ -d "$ENV" ] || conda create -y -p "$ENV" python=3.10
conda activate "$ENV"
pip install --upgrade pip
pip install git+https://github.com/huggingface/parler-tts.git soundfile pydub
conda install -y -c conda-forge ffmpeg
echo "[parler-env] DONE"

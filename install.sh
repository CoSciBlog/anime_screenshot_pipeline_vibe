#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python_cmd="${PYTHON:-python3}"
if ! command -v "${python_cmd}" >/dev/null 2>&1; then
  python_cmd="python"
fi

if [ ! -x "env/bin/python" ]; then
  printf 'Creating Python virtual environment in env...\n'
  "${python_cmd}" -m venv env
fi

env/bin/python -m pip install --upgrade pip uv
git submodule update --init --recursive

if command -v nvidia-smi >/dev/null 2>&1; then
  printf 'Installing CUDA 12.8 PyTorch wheels for NVIDIA GPUs...\n'
  env/bin/python -m uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
else
  printf 'NVIDIA GPU not detected. Installing CPU-compatible PyTorch wheels...\n'
  env/bin/python -m uv pip install torch torchvision torchaudio
fi

env/bin/python -m uv pip install -r requirements.txt
env/bin/python -m uv pip install -e waifuc
env/bin/python -m uv pip install -e .

printf 'Setup completed. Activate the environment with: source env/bin/activate\n'
printf 'Stage 1 additionally requires ffmpeg on PATH.\n'

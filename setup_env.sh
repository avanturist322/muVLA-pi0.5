#!/usr/bin/env bash
# Idempotent environment bootstrap for pi05-mem.
# All caches are pinned INSIDE the project root (NFS volume has ~880G free,
# while the container overlay is small and wiped between jobs).
set -euo pipefail

ROOT="${PI05_MEM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export UV_CACHE_DIR="$ROOT/.cache/uv"
export PIP_CACHE_DIR="$ROOT/.cache/pip"
export HF_HOME="$ROOT/.cache/hf"
export TRITON_CACHE_DIR="$ROOT/.cache/triton"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME" "$TRITON_CACHE_DIR" "$ROOT/.vendor"

UV="${UV:-uv}"   # any uv >= 0.4 will do; see https://docs.astral.sh/uv/
"$UV" --version

# --- lerobot source checkout (we fork pi05, so we need the source) ---
if [ ! -d "$ROOT/.vendor/lerobot/.git" ]; then
  git clone --depth 50 https://github.com/huggingface/lerobot.git "$ROOT/.vendor/lerobot"
fi
cd "$ROOT/.vendor/lerobot"
git log --oneline -1

# --- venv ---
cd "$ROOT"
# lerobot main requires Python >= 3.12; uv fetches a managed interpreter for us.
export UV_PYTHON_INSTALL_DIR="$ROOT/.cache/uv-python"
if [ ! -d "$ROOT/.venv" ]; then
  "$UV" venv --python 3.12 "$ROOT/.venv"
fi
export VIRTUAL_ENV="$ROOT/.venv"

# torch first (cu121 wheels match the driver on this node)
"$UV" pip install --python "$ROOT/.venv/bin/python" "torch==2.7.1" "torchvision==0.22.1" \
  --index-url https://download.pytorch.org/whl/cu126

# lerobot in editable mode with the pi0/pi05 extra
"$UV" pip install --python "$ROOT/.venv/bin/python" -e "$ROOT/.vendor/lerobot[pi]"

"$ROOT/.venv/bin/python" -c "
import torch, lerobot
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('lerobot', lerobot.__file__)
import lerobot.policies.pi05.modeling_pi05 as m
print('pi05 ok:', m.__file__)
"
echo SETUP_DONE

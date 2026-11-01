#!/usr/bin/env bash
# Build the standalone CPython3.12 LIBERO/VLA runtime from fixed public packages.
# Context: Policy, camera and physics imports must not depend on another project.
# Decision: An isolated virtualenv with the complete evaluated runtime pin set.
# Evidence: Both official policies and LIBERO import in an independent environment.
# Constraints: Linux x86_64 and a CUDA12.8-compatible driver; no checkpoint download here.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="$PWD/.cache/uv"
export HF_HOME="$PWD/.cache/huggingface"
export XDG_CACHE_HOME="$PWD/.cache"
export XDG_CONFIG_HOME="$PWD/.cache/config"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
PAI_PYTHON="${PAI_PYTHON:-/usr/bin/python3}"
PAI_RUNTIME_DIR="${PAI_RUNTIME_DIR:-$PWD/.venv-realtime}"
mkdir -p .tools "$UV_CACHE_DIR"
"$PAI_PYTHON" -c 'import sys; assert sys.version_info[:2]==(3,12), "CPython3.12 is required"'
if [[ ! -x .tools/uv ]]; then
    "$PAI_PYTHON" scripts/bootstrap_uv.py
fi
if [[ ! -x "$PAI_RUNTIME_DIR/bin/python" ]]; then
    .tools/uv venv --python "$PAI_PYTHON" "$PAI_RUNTIME_DIR"
fi
# Curated exact pins reproduce the evaluated runtime without unrelated optional extras.
.tools/uv --quiet pip install --python "$PAI_RUNTIME_DIR/bin/python" --no-deps -r requirements-realtime.lock
if [[ "${PAI_SKIP_ASSETS:-0}" != 1 ]]; then
    "$PAI_RUNTIME_DIR/bin/python" scripts/fetch_libero_assets.py
fi
MUJOCO_GL=egl "$PAI_RUNTIME_DIR/bin/python" scripts/verify_runtime.py

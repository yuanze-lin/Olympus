#!/usr/bin/env bash
# Install TRELLIS (v1) for native text-to-3D (<3D_gen_text> via TRELLIS-text-base).
#
# This is a separate checkout from TRELLIS.2 (scripts/install_trellis2.sh) --
# TRELLIS-text-base is only published for the original TRELLIS pipeline code.
# It compiles several CUDA extensions (nvdiffrast, diffoctreerast,
# diff-gaussian-rasterization) and needs NVIDIA's kaolin wheels. It is OPTIONAL
# -- without it <3D_gen_text> falls back to Shap-E, which works but produces
# noticeably worse meshes.
#
# Requirements: Linux, an NVIDIA GPU with >= 16GB, CUDA 12.4 toolchain.
#
# Usage:  bash scripts/install_trellis1.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party"
mkdir -p "$THIRD_PARTY"

if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "error: activate the olympus conda environment first" >&2
  exit 1
fi

if ! nvcc --version 2>/dev/null | grep -q "release 12.4"; then
  echo "[cuda] installing the CUDA 12.4 toolchain into $CONDA_PREFIX"
  conda install -y -p "$CONDA_PREFIX" -c nvidia/label/cuda-12.4.0 cuda-toolkit
fi
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version | tail -2

# --- TRELLIS (v1) --------------------------------------------------------
if [ ! -d "$THIRD_PARTY/TRELLIS/.git" ]; then
  echo "[clone] microsoft/TRELLIS -> third_party/TRELLIS"
  git clone --recursive https://github.com/microsoft/TRELLIS.git "$THIRD_PARTY/TRELLIS"
fi

cd "$THIRD_PARTY/TRELLIS"
export MAX_JOBS="${MAX_JOBS:-16}"
export ATTN_BACKEND=xformers
export SPCONV_ALGO=native

# kaolin ships prebuilt wheels for a matrix of (torch, cuda) versions -- far
# faster and more reliable than a source build.
echo "[build] installing kaolin (prebuilt wheel for torch 2.6.0+cu124)"
pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.6.0_cu124.html \
  || pip install kaolin

# The remaining extensions (nvdiffrast, diffoctreerast, mip-splatting's
# diff-gaussian-rasterization, spconv) are pulled in by TRELLIS's own setup
# script. --no-build-isolation is required: these are CUDA extensions that
# import torch during their own build, which pip's isolated build env hides
# unless told otherwise.
echo "[build] compiling TRELLIS extensions (this takes a while)"
# shellcheck disable=SC1091
. ./setup.sh --basic --xformers --spconv --diffoctreerast --mipgaussian --nvdiffrast

cd "$ROOT"
python - <<'PY'
import sys, os
sys.path.insert(0, "third_party/TRELLIS")
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")
try:
    import kaolin  # noqa: F401
    from trellis.pipelines import TrellisTextTo3DPipeline  # noqa: F401
    print("\nTRELLIS (v1) installed successfully -- <3D_gen_text> will use TRELLIS-text-base.")
except Exception as exc:
    print(f"\nTRELLIS import FAILED: {type(exc).__name__}: {exc}")
    print("<3D_gen_text> will fall back to Shap-E. Re-run this script after "
          "fixing the build error above.")
    raise SystemExit(1)
PY

echo
echo "Weights download automatically on first use (microsoft/TRELLIS-text-base),"
echo "or pre-fetch them now with:"
echo "  python -c \"from huggingface_hub import snapshot_download as d; d('microsoft/TRELLIS-text-base')\""

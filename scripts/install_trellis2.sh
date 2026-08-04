#!/usr/bin/env bash
# Install TRELLIS.2 for high-quality image-to-3D (<3D_gen_image>, <3D_gen_text>).
#
# This is the heaviest dependency in the project: TRELLIS.2 compiles several CUDA
# extensions (flash-attn, nvdiffrast, nvdiffrec, cumesh, o-voxel, flexgemm) and
# the build takes a while. It is OPTIONAL -- without it the 3D tokens fall back to
# Shap-E, which works but produces noticeably worse meshes.
#
# Requirements: Linux, an NVIDIA GPU with >= 24GB, and the CUDA 12.4 toolchain.
# If your system has no CUDA 12.4, this script installs it into the conda
# environment, so no root access or system change is needed.
#
# Usage:  bash scripts/install_trellis2.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party"
mkdir -p "$THIRD_PARTY"

if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "error: activate the olympus conda environment first" >&2
  exit 1
fi

# --- CUDA 12.4 toolchain -----------------------------------------------------
# TRELLIS.2 expects nvcc 12.4 (matching torch 2.6.0+cu124). Install it into the
# active environment if the system does not provide it.
if ! nvcc --version 2>/dev/null | grep -q "release 12.4"; then
  echo "[cuda] installing the CUDA 12.4 toolchain into $CONDA_PREFIX"
  conda install -y -p "$CONDA_PREFIX" -c nvidia/label/cuda-12.4.0 cuda-toolkit
fi
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version | tail -2

# --- TRELLIS.2 ---------------------------------------------------------------
if [ ! -d "$THIRD_PARTY/TRELLIS.2/.git" ]; then
  echo "[clone] microsoft/TRELLIS.2 -> third_party/TRELLIS.2"
  git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive \
    "$THIRD_PARTY/TRELLIS.2"
fi

cd "$THIRD_PARTY/TRELLIS.2"
export MAX_JOBS="${MAX_JOBS:-16}"

# NOTE: --new-env is deliberately omitted so everything lands in the existing
# olympus environment -- Olympus keeps the router and all specialists together.
echo "[build] compiling TRELLIS.2 extensions (this takes a while)"
# shellcheck disable=SC1091
. ./setup.sh --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm

cd "$ROOT"
python - <<'PY'
import sys
sys.path.insert(0, "third_party/TRELLIS.2")
try:
    import o_voxel  # noqa: F401
    from trellis2.pipelines import Trellis2ImageTo3DPipeline  # noqa: F401
    print("\nTRELLIS.2 installed successfully -- <3D_gen_image> will use it.")
except Exception as exc:
    print(f"\nTRELLIS.2 import FAILED: {type(exc).__name__}: {exc}")
    print("The 3D tokens will fall back to Shap-E. Re-run this script after "
          "fixing the build error above.")
    raise SystemExit(1)
PY

echo
echo "Weights download automatically on first use (microsoft/TRELLIS.2-4B),"
echo "or pre-fetch them now with:"
echo "  python -c \"from huggingface_hub import snapshot_download as d; d('microsoft/TRELLIS.2-4B')\""
echo
echo "NOTE: TRELLIS.2 also loads two gated repos by name at runtime:"
echo "  - facebook/dinov3-vitl16-pretrain-lvd1689m (image conditioning)"
echo "  - briaai/RMBG-2.0 (background removal)"
echo "Accept their licenses on Hugging Face while logged in (huggingface-cli login)."
echo "If your request is rejected, obtain the weights locally (see"
echo "https://github.com/microsoft/TRELLIS.2/issues/38) and point at them:"
echo "  DINO_MODEL_PATH=/path/to/dinov3 SEG_MODEL_PATH=/path/to/rmbg python run_tools.py ..."

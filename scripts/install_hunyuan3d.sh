#!/usr/bin/env bash
# Install Hunyuan3D-2 for image-to-3D (<3D_gen_image> fallback when TRELLIS.2
# is unavailable -- e.g. the DINOv3/RMBG gated-repo access hasn't been
# granted yet). Fully open, no gated repos, easier to stand up than TRELLIS.2.
#
# The shape-generation stage works out of the box on plain torch/diffusers.
# The optional texture-painting stage (Hunyuan3D-Paint) needs a custom
# rasterizer extension; when it fails to build, Hunyuan3DBackend automatically
# falls back to returning an untextured mesh rather than erroring out.
#
# Usage:  bash scripts/install_hunyuan3d.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party"
mkdir -p "$THIRD_PARTY"

if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "error: activate the olympus conda environment first" >&2
  exit 1
fi

if [ ! -d "$THIRD_PARTY/Hunyuan3D-2/.git" ]; then
  echo "[clone] tencent/Hunyuan3D-2 -> third_party/Hunyuan3D-2"
  git clone https://github.com/Tencent/Hunyuan3D-2.git "$THIRD_PARTY/Hunyuan3D-2"
fi

cd "$THIRD_PARTY/Hunyuan3D-2"
pip install -q -r requirements.txt

# Texture-painting rasterizer -- optional, best-effort. If this fails, the
# shape-generation stage still works; Hunyuan3DBackend just returns an
# untextured mesh in that case.
if [ -d "hy3dgen/texgen/custom_rasterizer" ]; then
  echo "[build] custom_rasterizer (optional texture stage)"
  (cd hy3dgen/texgen/custom_rasterizer && pip install -e . --no-build-isolation) \
    || echo "custom_rasterizer build failed -- texture stage will be skipped at runtime"
fi
if [ -d "hy3dgen/texgen/differentiable_renderer" ]; then
  echo "[build] differentiable_renderer (optional texture stage)"
  (cd hy3dgen/texgen/differentiable_renderer && bash compile_mesh_painter.sh) \
    || echo "differentiable_renderer build failed -- texture stage will be skipped at runtime"
fi

cd "$ROOT"
python - <<'PY'
import sys
sys.path.insert(0, "third_party/Hunyuan3D-2")
try:
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
    print("\nHunyuan3D-2 installed successfully -- <3D_gen_image> can use it as a fallback.")
except Exception as exc:
    print(f"\nHunyuan3D-2 import FAILED: {type(exc).__name__}: {exc}")
    raise SystemExit(1)
PY

echo
echo "Weights download automatically on first use (tencent/Hunyuan3D-2)."
echo "Licence: tencent-hunyuan-community (not MIT) -- see"
echo "  https://huggingface.co/tencent/Hunyuan3D-2/blob/main/LICENSE.txt"

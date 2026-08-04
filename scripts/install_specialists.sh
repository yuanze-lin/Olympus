#!/usr/bin/env bash
# Clone the specialist repositories that are not pip-installable.
#
# Everything else (Stable Diffusion XL, InstructPix2Pix, ControlNet, CogVideoX,
# Shap-E, SegFormer, GroundingDINO, Depth Anything V2, Swin2SR, SAM) is pulled
# straight from the Hugging Face Hub on first use, so only these two need a clone.
#
# Usage:  bash scripts/install_specialists.sh [--with-triposr]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party"
mkdir -p "$THIRD_PARTY"

# transformers >= 4.49 imports timm.data.ImageNetInfo (added in timm 1.0) via its
# timm_wrapper model; without it <image_ground> and the InstructIR tokens fail to
# import. controlnet_aux 0.0.9 declares timm<=0.6.7, so the two cannot be pinned
# in the same requirements file -- pip errors with ResolutionImpossible. That pin
# is stale, so upgrade timm here, after controlnet_aux is already installed.
echo "[deps] upgrading timm to 1.0.15 (overrides controlnet_aux's stale pin)"
pip install -q "timm==1.0.15" 2>&1 | grep -v "dependency resolver" || true

clone() {  # clone <url> <dir> [ref]
  local url="$1" dir="$2" ref="${3:-}"
  if [ -d "$THIRD_PARTY/$dir/.git" ]; then
    echo "[skip] $dir already cloned"
    return
  fi
  echo "[clone] $url -> third_party/$dir"
  git clone --depth 1 "$url" "$THIRD_PARTY/$dir"
  if [ -n "$ref" ]; then
    git -C "$THIRD_PARTY/$dir" fetch --depth 1 origin "$ref"
    git -C "$THIRD_PARTY/$dir" checkout "$ref"
  fi
}

# InstructIR -- <image_deblur>, <image_denoise>, <image_derain>  (Table 9)
clone https://github.com/mv-lab/InstructIR.git InstructIR

# TripoSR -- optional higher-quality <3D_gen_image>; without it Shap-E is used.
WITH_TRIPOSR=0
for arg in "$@"; do
  [ "$arg" = "--with-triposr" ] && WITH_TRIPOSR=1
done
if [ "$WITH_TRIPOSR" = "1" ]; then
  clone https://github.com/VAST-AI-Research/TripoSR.git TripoSR
  pip install -q torchmcubes || echo "[warn] torchmcubes failed to build; TripoSR will be skipped"
fi

cat <<EOF

Specialist repositories installed under: $THIRD_PARTY

  InstructIR  -> <image_deblur> / <image_denoise> / <image_derain>
$([ "$WITH_TRIPOSR" = "1" ] && echo "  TripoSR     -> <3D_gen_image> (preferred over Shap-E)")

Hub-hosted specialists download automatically on first use.
Point HF_HOME at a large disk if your home directory is small, e.g.
  export HF_HOME=/path/to/big/disk/hf_cache

Verify the install with:
  python run_tools.py --list-tokens
EOF

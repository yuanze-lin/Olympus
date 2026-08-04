#!/usr/bin/env bash
# One-shot installer for all three optional high-quality 3D backends:
#   - TRELLIS.2-4B       (<3D_gen_image>, primary)
#   - Hunyuan3D-2        (<3D_gen_image>, ungated fallback)
#   - TRELLIS-text-base  (<3D_gen_text>)
#
# Each sub-script is independent and safe to re-run; a failure in one does not
# stop the others, so you still end up with the best backend that could build
# on your machine. See scripts/install_trellis2.sh, scripts/install_trellis1.sh
# and scripts/install_hunyuan3d.sh for details on each.
#
# Usage:  bash scripts/install_3d.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "error: activate the olympus conda environment first" >&2
  exit 1
fi

status=0

echo "==> [1/3] TRELLIS.2-4B (<3D_gen_image>, primary)"
bash scripts/install_trellis2.sh || { echo "TRELLIS.2 install failed, continuing"; status=1; }

echo
echo "==> [2/3] Hunyuan3D-2 (<3D_gen_image>, ungated fallback)"
bash scripts/install_hunyuan3d.sh || { echo "Hunyuan3D-2 install failed, continuing"; status=1; }

echo
echo "==> [3/3] TRELLIS-text-base (<3D_gen_text>)"
bash scripts/install_trellis1.sh || { echo "TRELLIS (v1) install failed, continuing"; status=1; }

echo
echo "==> Resolving backends actually available"
python -c "
import olympus_tools.backends as b
b.resolve_3d_backends()
from olympus_tools import tokens
print('3D_gen_image backend:', tokens.get_spec('3D_gen_image').backend)
print('3D_gen_text  backend:', tokens.get_spec('3D_gen_text').backend)
"

exit $status

"""Specialist backends for the Olympus routing tokens.

Importing this package registers every backend in
:data:`olympus_tools.backends.base._REGISTRY`.
"""

from .base import Backend, ModelHost, available_backends, register  # noqa: F401
from . import sota  # noqa: F401  (current SOTA defaults)
from . import threed_sota  # noqa: F401  (Hunyuan3D-2, TRELLIS text-to-3D)
from . import trellis2  # noqa: F401  (TRELLIS.2; needs gated DINOv3)
from . import diffusion  # noqa: F401  (Table 9 originals, still selectable)
from . import perception  # noqa: F401
from . import restoration  # noqa: F401
from . import threed  # noqa: F401
from . import rvos  # noqa: F401


def resolve_3d_backends() -> None:
    """Downgrade the 3D tokens when their backends are unavailable.

    ``<3D_gen_image>`` prefers TRELLIS.2, then Hunyuan3D-2 (ungated, no weight
    gate to clear), then Shap-E. ``<3D_gen_text>`` prefers TRELLIS-text-base,
    then Shap-E. A run should still produce a mesh rather than fail outright, but
    the quality gap is large, so each downgrade is reported.
    """
    from .threed_sota import hunyuan3d_available, trellis_text_available
    from .trellis2 import trellis2_available
    from .. import tokens as _tokens

    downgrade, notes = {}, []

    if not trellis2_available():
        if hunyuan3d_available():
            downgrade["3D_gen_image"] = "hunyuan3d"
            notes.append("<3D_gen_image>: TRELLIS.2 unavailable -> Hunyuan3D-2")
        else:
            downgrade["3D_gen_image"] = "image_to_3d"
            notes.append("<3D_gen_image>: no high-quality backend -> Shap-E")

    if not trellis_text_available():
        downgrade["3D_gen_text"] = "text_to_3d"
        notes.append("<3D_gen_text>: TRELLIS-text unavailable -> Shap-E")

    if downgrade:
        for n in notes:
            print(f"[olympus_tools] {n}")
        print("[olympus_tools] run scripts/install_3d.sh for better meshes.")
        _tokens.override_backends(downgrade)


__all__ = ["Backend", "ModelHost", "available_backends", "register",
           "resolve_3d_backends"]

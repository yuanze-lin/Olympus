"""Current-generation 3D backends: Hunyuan3D-2 and TRELLIS text-to-3D.

Shap-E (the previous default) produces meshes too coarse to be useful, so the 3D
tokens now use:

* ``<3D_gen_image>`` -> Hunyuan3D-2 (tencent/Hunyuan3D-2), image-to-3D
* ``<3D_gen_text>``  -> TRELLIS-text-base (microsoft/TRELLIS-text-base)

Both are ungated. TRELLIS.2-4B is stronger still and is implemented in
:mod:`olympus_tools.backends.trellis2`, but its image encoder
(``facebook/dinov3-...``) is a gated repo, so it cannot be the default.
"""

import os
import sys
from typing import Optional

import torch
from PIL import Image

from .base import Backend, register

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _third_party(name: str, env_var: str) -> Optional[str]:
    cand = os.environ.get(env_var)
    if cand and os.path.isdir(cand):
        return cand
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cand = os.path.join(here, "third_party", name)
    return cand if os.path.isdir(cand) else None


def hunyuan3d_available() -> bool:
    repo = _third_party("Hunyuan3D-2", "OLYMPUS_HUNYUAN3D_DIR")
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401

        return True
    except Exception:
        return False


def trellis_text_available() -> bool:
    repo = _third_party("TRELLIS", "OLYMPUS_TRELLIS_DIR")
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)
    os.environ.setdefault("ATTN_BACKEND", "xformers")
    os.environ.setdefault("SPCONV_ALGO", "native")
    try:
        from trellis.pipelines import TrellisTextTo3DPipeline  # noqa: F401

        return True
    except Exception:
        return False


@register("hunyuan3d")
class Hunyuan3DBackend(Backend):
    """<3D_gen_image> -- Hunyuan3D-2 image-to-3D.

    Runs the shape pipeline (Hunyuan3D-DiT) and, when the optional texture
    extensions are built, the paint pipeline (Hunyuan3D-Paint) as well.

    Licence note: tencent-hunyuan-community, not MIT -- see
    https://huggingface.co/tencent/Hunyuan3D-2/blob/main/LICENSE.txt
    """

    default_model_id = "tencent/Hunyuan3D-2"
    substitution = "Wonder3D -> Hunyuan3D-2"

    def load(self):
        repo = _third_party("Hunyuan3D-2", "OLYMPUS_HUNYUAN3D_DIR")
        if repo and repo not in sys.path:
            sys.path.insert(0, repo)
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        self.pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(self.model_id)
        self._painter = None
        self._rembg = None

    def _remove_background(self, img: Image.Image) -> Image.Image:
        """Hunyuan3D expects a foreground-isolated RGBA image."""
        if img.mode == "RGBA":
            return img
        try:
            from hy3dgen.rembg import BackgroundRemover

            if self._rembg is None:
                self._rembg = BackgroundRemover()
            return self._rembg(img)
        except Exception:
            try:
                import rembg

                return rembg.remove(img)
            except Exception:
                return img

    def _texture(self, mesh, img: Image.Image):
        """Optional PBR texturing; needs the custom rasterizer extensions."""
        try:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline

            if self._painter is None:
                self._painter = Hunyuan3DPaintPipeline.from_pretrained(self.model_id)
            return self._painter(mesh, image=img), True
        except Exception as exc:
            print(f"[3D] Hunyuan3D texture stage unavailable ({type(exc).__name__}); "
                  f"returning untextured geometry. Build the texture extensions in "
                  f"third_party/Hunyuan3D-2 to enable it.")
            return mesh, False

    def run(self, prompt, input_path, out_stem, step=None, **kw):
        if not input_path:
            raise ValueError("<3D_gen_image> needs an input image "
                             "(chain it after an image step or pass --input-image)")
        img = self._remove_background(Image.open(input_path).convert("RGB"))

        mesh = self.pipe(
            image=img,
            num_inference_steps=kw.get("steps", 30),
            octree_resolution=kw.get("octree_resolution", 380),
            guidance_scale=kw.get("guidance", 5.5),
            generator=torch.manual_seed(kw.get("seed", 0)),
        )[0]

        textured = False
        if kw.get("texture", True):
            mesh, textured = self._texture(mesh, img)

        out = f"{out_stem}.glb"
        mesh.export(out)
        return {"mesh": out, "glb": out, "model": self.model_id,
                "textured": textured}


@register("trellis_text")
class TrellisTextTo3DBackend(Backend):
    """<3D_gen_text> -- TRELLIS-text-base, native text-to-3D.

    Unlike the image-lifting route this is conditioned on text directly, using
    CLIP as the text encoder (ungated).
    """

    default_model_id = "microsoft/TRELLIS-text-base"
    substitution = "LGM -> TRELLIS-text-base"

    def load(self):
        repo = _third_party("TRELLIS", "OLYMPUS_TRELLIS_DIR")
        if repo and repo not in sys.path:
            sys.path.insert(0, repo)
        # TRELLIS reads these at import time.
        os.environ.setdefault("ATTN_BACKEND", "xformers")
        os.environ.setdefault("SPCONV_ALGO", "native")
        from trellis.pipelines import TrellisTextTo3DPipeline

        self.pipe = TrellisTextTo3DPipeline.from_pretrained(self.model_id)
        self.pipe.cuda()

    def run(self, prompt, input_path, out_stem, step=None, **kw):
        from trellis.utils import postprocessing_utils

        outputs = self.pipe.run(
            prompt,
            seed=kw.get("seed", 0),
            sparse_structure_sampler_params={
                "steps": kw.get("steps", 25), "cfg_strength": 7.5},
            slat_sampler_params={"steps": kw.get("steps", 25), "cfg_strength": 7.5},
        )
        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=kw.get("simplify", 0.95),
            texture_size=kw.get("texture_size", 1024),
        )
        out = f"{out_stem}.glb"
        glb.export(out)
        return {"mesh": out, "glb": out, "model": self.model_id}

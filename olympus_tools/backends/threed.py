"""3D generation specialists.

Table 9 assigns LGM to ``<3D_gen_text>`` and Wonder3D to ``<3D_gen_image>``. Both
need bespoke CUDA rasterisers that frequently fail to build, which would defeat
the purpose of this integration, so the defaults are Shap-E pipelines that ship
inside ``diffusers`` and need no compilation. If ``third_party/TripoSR`` is
present it is preferred for image-to-3D because it produces a far better mesh.

Every backend writes a real mesh (``.ply`` plus ``.glb``) and a turntable preview,
so ``<3D_gen_image>`` yields an artifact you can open in any 3D viewer.
"""

import os
import sys
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .base import Backend, register, hf_kwargs


def _ply_to_glb(ply_path: str) -> Optional[str]:
    """Convert a PLY to GLB so the mesh opens in standard viewers."""
    try:
        import trimesh

        mesh = trimesh.load(ply_path)
        glb = os.path.splitext(ply_path)[0] + ".glb"
        mesh.export(glb)
        return glb
    except Exception:
        return None


def _turntable(ply_path: str, out_gif: str, frames: int = 24) -> Optional[str]:
    """Render a simple turntable preview of the mesh (best effort)."""
    try:
        import imageio
        import trimesh

        mesh = trimesh.load(ply_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        imgs = []
        scene = mesh.scene()
        for i in range(frames):
            scene.set_camera(angles=(0, np.radians(360 * i / frames), 0))
            imgs.append(imageio.v3.imread(scene.save_image(resolution=(256, 256))))
        imageio.mimsave(out_gif, imgs, fps=12, loop=0)
        return out_gif
    except Exception:
        return None


class _ShapEBase(Backend):
    def _finish(self, mesh, out_stem):
        from diffusers.utils import export_to_ply

        ply = f"{out_stem}.ply"
        export_to_ply(mesh, ply)
        result = {"mesh": ply}
        glb = _ply_to_glb(ply)
        if glb:
            result["glb"] = glb
        gif = _turntable(ply, f"{out_stem}_turntable.gif")
        if gif:
            result["preview"] = gif
        return result


@register("text_to_3d")
class TextTo3DBackend(_ShapEBase):
    """<3D_gen_text> -- text-to-3D generation.

    Table 9 names LGM; Shap-E is the zero-build default.
    """

    default_model_id = "openai/shap-e"
    substitution = "LGM -> Shap-E (diffusers, no CUDA build required)"

    def load(self):
        from diffusers import ShapEPipeline

        # Shap-E's prior runs its projection layers in fp32; loading the pipeline
        # in fp16 raises "mat1 and mat2 have the same dtype" at inference. The
        # model is small, so fp32 costs little.
        self.dtype = torch.float32
        self.pipe = ShapEPipeline.from_pretrained(self.model_id,
                                                  **hf_kwargs(self.dtype)).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    def run(self, prompt, input_path, out_stem, step=None, **kw):
        g = torch.Generator(device=self.device).manual_seed(kw.get("seed", 0))
        mesh = self.pipe(
            prompt,
            guidance_scale=kw.get("guidance", 15.0),
            num_inference_steps=kw.get("steps", 64),
            frame_size=kw.get("frame_size", 256),
            output_type="mesh",
            generator=g,
        ).images[0]
        return self._finish(mesh, out_stem)


@register("image_to_3d")
class ImageTo3DBackend(_ShapEBase):
    """<3D_gen_image> -- image-to-3D generation.

    Table 9 names Wonder3D. TripoSR is used when present (better meshes);
    otherwise Shap-E image-to-3D, which needs no compilation.
    """

    default_model_id = "openai/shap-e-img2img"
    substitution = "Wonder3D -> TripoSR if installed, else Shap-E img2img"

    def _triposr_dir(self):
        cand = os.environ.get("OLYMPUS_TRIPOSR_DIR")
        if cand and os.path.isdir(cand):
            return cand
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cand = os.path.join(here, "third_party", "TripoSR")
        return cand if os.path.isdir(cand) else None

    def load(self):
        self._mode = "shap-e"
        repo = self._triposr_dir()
        if repo:
            try:
                if repo not in sys.path:
                    sys.path.insert(0, repo)
                from tsr.system import TSR

                self.model = TSR.from_pretrained("stabilityai/TripoSR",
                                                 config_name="config.yaml",
                                                 weight_name="model.ckpt")
                self.model.renderer.set_chunk_size(8192)
                self.model.to(self.device)
                self._mode = "triposr"
                return
            except Exception as exc:  # pragma: no cover - optional path
                print(f"[image_to_3d] TripoSR unavailable ({exc}); using Shap-E")

        from diffusers import ShapEImg2ImgPipeline

        self.pipe = ShapEImg2ImgPipeline.from_pretrained(
            self.model_id, **hf_kwargs(self.dtype)).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    def run(self, prompt, input_path, out_stem, step=None, **kw):
        if not input_path:
            raise ValueError("<3D_gen_image> needs an input image "
                             "(chain it after an image step or pass --input-image)")
        img = Image.open(input_path).convert("RGB")

        if self._mode == "triposr":
            from tsr.utils import remove_background, resize_foreground
            import rembg

            session = rembg.new_session()
            clean = resize_foreground(remove_background(img, session), 0.85)
            with torch.no_grad():
                codes = self.model([clean], device=self.device)
                mesh = self.model.extract_mesh(codes, resolution=kw.get("mc_res", 256))[0]
            obj = f"{out_stem}.obj"
            mesh.export(obj)
            result = {"mesh": obj}
            glb = _ply_to_glb(obj)
            if glb:
                result["glb"] = glb
            gif = _turntable(obj, f"{out_stem}_turntable.gif")
            if gif:
                result["preview"] = gif
            result["backend"] = "TripoSR"
            return result

        img = img.resize((256, 256))
        g = torch.Generator(device=self.device).manual_seed(kw.get("seed", 0))
        mesh = self.pipe(
            img,
            guidance_scale=kw.get("guidance", 3.0),
            num_inference_steps=kw.get("steps", 64),
            frame_size=kw.get("frame_size", 256),
            output_type="mesh",
            generator=g,
        ).images[0]
        out = self._finish(mesh, out_stem)
        out["backend"] = "Shap-E"
        return out

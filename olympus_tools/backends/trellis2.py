"""3D generation with TRELLIS.2 (microsoft/TRELLIS.2-4B).

Replaces Shap-E, whose meshes are too coarse to be useful. TRELLIS.2 is a 4B
flow-matching model over a sparse "O-Voxel" representation and produces textured
meshes with PBR materials at up to 1536 voxel resolution.

It needs compiled CUDA extensions (see scripts/install_trellis2.sh). When they are
absent the backends in :mod:`olympus_tools.backends.threed` are used instead, so
``<3D_gen_image>`` still works -- at lower quality.
"""

import os
import sys
from typing import Optional

import torch
from PIL import Image

from .base import Backend, register, hf_kwargs

# Keeps peak memory down on a single GPU.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")


def _select_attn_backend() -> str:
    """Prefer flash-attn, fall back to xformers.

    TRELLIS.2 defaults to flash-attn, which frequently fails to build (pip build
    isolation cannot see torch). xformers is a supported alternative and is far
    easier to install, so it is selected automatically when flash-attn is absent
    rather than letting the pipeline fail at import time.
    """
    if "ATTN_BACKEND" in os.environ:
        return os.environ["ATTN_BACKEND"]
    try:
        import flash_attn  # noqa: F401

        backend = "flash-attn"
    except Exception:
        backend = "xformers"
    os.environ["ATTN_BACKEND"] = backend
    return backend


def _patch_model_name(cls, path: str, label: str) -> None:
    """Retry ``cls.__init__`` with ``path`` in place of the Hub repo name.

    Re-running the real ``__init__`` rather than reimplementing it matters: both
    of these classes set several attributes besides ``self.model`` (``eval()``,
    transforms, image size), so a hand-rolled fallback constructs an object that
    only fails later.
    """
    orig = cls.__init__

    def _init(self, *a, **kw):
        try:
            return orig(self, *a, **kw)
        except Exception:
            print(f"[3D] {label}: loading from {path}")
            if a:
                a = (path,) + tuple(a[1:])
            else:
                kw = dict(kw, model_name=path)
            return orig(self, *a, **kw)

    cls.__init__ = _init


def _apply_local_weight_overrides() -> None:
    """Let DINO_MODEL_PATH / SEG_MODEL_PATH point at locally-stored weights.

    TRELLIS.2 loads two gated repos by name: facebook/dinov3-vitl16-pretrain-lvd1689m
    (image conditioning) and briaai/RMBG-2.0 (background removal). Users whose
    access request was rejected, or who run offline, can set these environment
    variables to a directory instead::

        DINO_MODEL_PATH=/path/to/dinov3 SEG_MODEL_PATH=/path/to/rmbg \\
            python run_tools.py ...

    Patched at runtime so third_party/TRELLIS.2 stays an unmodified checkout.
    """
    dino_path = os.environ.get("DINO_MODEL_PATH")
    seg_path = os.environ.get("SEG_MODEL_PATH")
    if not dino_path and not seg_path:
        return

    if dino_path:
        try:
            from trellis2.modules import image_feature_extractor as _ife

            _patch_model_name(_ife.DinoV3FeatureExtractor, dino_path, "DINOv3")
        except Exception as exc:
            print(f"[3D] could not apply DINO_MODEL_PATH override: {exc}")

    if seg_path:
        try:
            from trellis2.pipelines import rembg as _rembg

            # rembg/__init__.py star-imports the class over the submodule of the
            # same name, so accept either layout.
            target = getattr(_rembg, "BiRefNet")
            if not isinstance(target, type):
                target = target.BiRefNet
            _patch_model_name(target, seg_path, "RMBG")
        except Exception as exc:
            print(f"[3D] could not apply SEG_MODEL_PATH override: {exc}")


def _trellis2_dir() -> Optional[str]:
    cand = os.environ.get("OLYMPUS_TRELLIS2_DIR")
    if cand and os.path.isdir(cand):
        return cand
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cand = os.path.join(here, "third_party", "TRELLIS.2")
    return cand if os.path.isdir(cand) else None


_GATED_REPOS = (
    "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "briaai/RMBG-2.0",
)


def _gated_weights_reachable() -> bool:
    """True if the two gated repos TRELLIS.2 needs are actually usable.

    A repo is usable if either (a) the caller's HF token has been granted
    access -- checked with a cheap metadata call, no download -- or (b) a
    local override path is set (DINO_MODEL_PATH / SEG_MODEL_PATH), which lets
    :func:`_apply_local_weight_overrides` load it from disk instead. This is
    a preflight check so unavailability is caught *before* the pipeline is
    built (and reported as a clean downgrade), rather than surfacing as a
    raw 401/403 HTTPError mid-run.
    """
    if os.environ.get("DINO_MODEL_PATH") and os.environ.get("SEG_MODEL_PATH"):
        return True
    try:
        from huggingface_hub import hf_hub_download

        for repo_id in _GATED_REPOS:
            if repo_id == "facebook/dinov3-vitl16-pretrain-lvd1689m" and os.environ.get("DINO_MODEL_PATH"):
                continue
            if repo_id == "briaai/RMBG-2.0" and os.environ.get("SEG_MODEL_PATH"):
                continue
            try:
                # model_info() alone is not a reliable probe: HF returns repo
                # metadata for gated repos even when the caller's token has not
                # been granted access. Only an actual file fetch triggers the
                # 401/403 gate check, so request the smallest file that exists
                # on every one of these repos.
                hf_hub_download(repo_id, filename="config.json")
            except Exception:
                return False
        return True
    except Exception:
        # huggingface_hub missing or offline -- assume unreachable rather than
        # let an ImportError masquerade as "available".
        return False


def trellis2_available() -> bool:
    """True when TRELLIS.2 imports cleanly AND its gated weights are reachable
    (via Hub access or a local override path)."""
    repo = _trellis2_dir()
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)
    _select_attn_backend()
    try:
        import o_voxel  # noqa: F401
        from trellis2.pipelines import Trellis2ImageTo3DPipeline  # noqa: F401
    except Exception:
        return False
    return _gated_weights_reachable()


@register("trellis2")
class Trellis2Backend(Backend):
    """<3D_gen_image> -- TRELLIS.2-4B image-to-3D.

    Emits a textured GLB (PBR materials) plus the raw mesh. Falls back to the
    Shap-E/TripoSR path when the CUDA extensions are not built.
    """

    default_model_id = "microsoft/TRELLIS.2-4B"

    def load(self):
        repo = _trellis2_dir()
        if repo and repo not in sys.path:
            sys.path.insert(0, repo)
        backend = _select_attn_backend()
        print(f"[3D] TRELLIS.2 attention backend: {backend}")
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        _apply_local_weight_overrides()

        self.pipe = Trellis2ImageTo3DPipeline.from_pretrained(self.model_id)
        self.pipe.cuda()

    def _to_glb(self, mesh, out_stem: str, **kw) -> str:
        import o_voxel

        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=kw.get("decimation_target", 1000000),
            texture_size=kw.get("texture_size", 2048),
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=False,
        )
        out = f"{out_stem}.glb"
        glb.export(out, extension_webp=True)
        return out

    def _render_videos(self, mesh, out_stem: str) -> dict:
        """Turntable renders; best effort, needs the repo's HDRI asset.

        ``render_video`` returns every PBR channel in one pass, so both the
        beauty render and the channel sheet are essentially free once it has run.
        """
        try:
            import cv2
            import imageio
            from trellis2.renderers import EnvMap
            from trellis2.utils import render_utils

            hdri = os.path.join(_trellis2_dir() or "", "assets", "hdri", "forest.exr")
            envmap = None
            if os.path.exists(hdri):
                img = cv2.cvtColor(cv2.imread(hdri, cv2.IMREAD_UNCHANGED),
                                   cv2.COLOR_BGR2RGB)
                envmap = EnvMap(torch.tensor(img, dtype=torch.float32, device="cuda"))
            rendered = render_utils.render_video(mesh, envmap=envmap)

            out = {}
            # The lit turntable on its own: what you actually want to look at.
            # Keyed "render" rather than "video" so it stays an artifact and does
            # not become chainable input for a downstream video step.
            video = f"{out_stem}.mp4"
            imageio.mimsave(video, rendered["shaded"], fps=15)
            out["render"] = video
            # Shaded + normal + base colour + metallic + roughness + alpha tiled
            # into one frame, as in the model card's example.
            pbr = f"{out_stem}_pbr.mp4"
            imageio.mimsave(pbr, render_utils.make_pbr_vis_frames(rendered), fps=15)
            out["render_pbr"] = pbr
            return out
        except Exception:
            return {}

    def run(self, prompt, input_path, out_stem, step=None, **kw):
        if not input_path:
            raise ValueError("<3D_gen_image> needs an input image "
                             "(chain it after an image step or pass --input-image)")
        image = Image.open(input_path).convert("RGB")
        mesh = self.pipe.run(image, seed=kw.get("seed", 0))[0]
        mesh.simplify(kw.get("max_faces", 16777216))  # nvdiffrast limit

        result = {"mesh": self._to_glb(mesh, out_stem, **kw)}
        result["glb"] = result["mesh"]
        result.update(self._render_videos(mesh, out_stem))
        result["model"] = self.model_id
        return result


@register("text_to_3d_via_image")
class TextTo3DViaImageBackend(Backend):
    """<3D_gen_text> -- text-to-3D as text -> image -> TRELLIS.2.

    Direct text-to-3D models (Shap-E, LGM) are much weaker than running a strong
    text-to-image model and lifting the result, so this composes the configured
    ``<image_gen>`` backend with TRELLIS.2. Each stage is freed before the next
    loads, so peak VRAM stays at one model.
    """

    default_model_id = "microsoft/TRELLIS.2-4B"

    def load(self):
        # Sub-models are built lazily inside run() so only one is resident.
        self._t2i_id = self.options.get("t2i_backend", "qwen_image")

    def run(self, prompt, input_path, out_stem, step=None, **kw):
        from .base import _REGISTRY

        # 1. text -> image
        if input_path:
            image_path = input_path
        else:
            t2i = _REGISTRY[self._t2i_id](device=self.device, dtype="fp16")
            t2i.ensure_loaded()
            try:
                image_path = t2i.run(prompt, None, f"{out_stem}_render",
                                     step=None, **kw)["image"]
            finally:
                t2i.unload()

        # 2. image -> 3D
        lifter = _REGISTRY["trellis2"](device=self.device, dtype=self.dtype)
        lifter.ensure_loaded()
        try:
            out = lifter.run(prompt, image_path, out_stem, step=step, **kw)
        finally:
            lifter.unload()
        out["intermediate_image"] = image_path
        return out

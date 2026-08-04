"""State-of-the-art specialist backends.

The models named in Table 9 of the paper date from 2023-2024. This module wires
the routing tokens to current open-weight models instead, while the Table 9
originals stay available in :mod:`olympus_tools.backends.diffusion` and can be
re-selected with ``--backend-model`` or ``--legacy-backends``.

| token           | SOTA default            | Table 9 original  |
|-----------------|-------------------------|-------------------|
| <image_gen>     | Qwen-Image              | Stable Diffusion XL |
| <image_edit>    | Qwen-Image-Edit-2509    | InstructPix2Pix   |
| <video_gen>     | Wan2.2-TI2V-5B          | CogVideoX         |
| <video_edit>    | Kiwi-Edit-5B-Instruct   | Text2Video-Zero   |
| <image_depth>   | Depth Anything 3        | Depth Anything V2 |
"""

import os
from typing import Optional

import torch
from PIL import Image

from .base import Backend, register, hf_kwargs


def _save_video(frames, path: str, fps: int = 16) -> str:
    from ..media import write_video

    return write_video(frames, path, fps=fps)


@register("qwen_image")
class QwenImageBackend(Backend):
    """<image_gen> -- Qwen-Image (20B MMDiT).

    Far stronger prompt following and text rendering than SDXL. Weights are ~40GB
    in bf16, so model CPU offload is enabled to keep this on a single GPU.
    """

    default_model_id = "Qwen/Qwen-Image"
    substitution = "Stable Diffusion XL -> Qwen-Image (SOTA open-weight T2I)"

    def load(self):
        from diffusers import DiffusionPipeline

        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.set_progress_bar_config(disable=True)

    def run(self, prompt, input_path, out_stem, step=None, **kw):
        g = torch.Generator(device="cpu").manual_seed(kw.get("seed", 0))
        aspect = kw.get("size", (1328, 1328))
        image = self.pipe(
            prompt=prompt,
            negative_prompt=kw.get("negative_prompt", " "),
            width=aspect[0], height=aspect[1],
            num_inference_steps=kw.get("steps", 30),
            true_cfg_scale=kw.get("guidance", 4.0),
            generator=g,
        ).images[0]
        out = f"{out_stem}.png"
        image.save(out)
        return {"image": out, "model": self.model_id}


@register("qwen_image_edit")
class QwenImageEditBackend(Backend):
    """<image_edit> -- Qwen-Image-Edit-2509.

    Instruction-guided editing with far better identity/background preservation
    than InstructPix2Pix.
    """

    default_model_id = "Qwen/Qwen-Image-Edit-2509"
    substitution = "InstructPix2Pix -> Qwen-Image-Edit-2509"

    def load(self):
        from diffusers import DiffusionPipeline

        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.set_progress_bar_config(disable=True)

    def run(self, prompt, input_path, out_stem, step=None, **kw):
        if not input_path:
            raise ValueError("<image_edit> needs an input image "
                             "(chain it after <image_gen> or pass --input-image)")
        img = Image.open(input_path).convert("RGB")
        g = torch.Generator(device="cpu").manual_seed(kw.get("seed", 0))
        try:
            out_img = self.pipe(
                image=[img], prompt=prompt,
                negative_prompt=kw.get("negative_prompt", " "),
                num_inference_steps=kw.get("steps", 40),
                true_cfg_scale=kw.get("guidance", 4.0),
                generator=g,
            ).images[0]
        except TypeError:
            # the 2508/base revision takes a bare image rather than a list
            out_img = self.pipe(
                image=img, prompt=prompt,
                num_inference_steps=kw.get("steps", 40),
                true_cfg_scale=kw.get("guidance", 4.0),
                generator=g,
            ).images[0]
        out = f"{out_stem}.png"
        out_img.save(out)
        return {"image": out, "model": self.model_id}


@register("wan_video")
class WanVideoBackend(Backend):
    """<video_gen> -- Wan2.2-TI2V-5B.

    Replaces CogVideoX: 720p @ 24fps from a 5B model that fits one GPU. Accepts an
    optional chained image, in which case it runs image-to-video.
    """

    default_model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    substitution = "CogVideoX -> Wan2.2-TI2V-5B"

    def load(self):
        from diffusers import AutoencoderKLWan, WanPipeline

        vae = AutoencoderKLWan.from_pretrained(self.model_id, subfolder="vae",
                                               torch_dtype=torch.float32)
        self.pipe = WanPipeline.from_pretrained(self.model_id, vae=vae,
                                                torch_dtype=torch.bfloat16)
        self.pipe.enable_model_cpu_offload()
        self.pipe.set_progress_bar_config(disable=True)

    def run(self, prompt, input_path, out_stem, step=None, **kw):
        g = torch.Generator(device="cpu").manual_seed(kw.get("seed", 0))
        frames = self.pipe(
            prompt=prompt,
            negative_prompt=kw.get("negative_prompt",
                                   "low quality, blurry, distorted, watermark"),
            height=kw.get("height", 704), width=kw.get("width", 1280),
            num_frames=kw.get("num_frames", 49),
            num_inference_steps=kw.get("steps", 40),
            guidance_scale=kw.get("guidance", 5.0),
            generator=g,
        ).frames[0]
        out = f"{out_stem}.mp4"
        _save_video(frames, out, fps=kw.get("fps", 24))
        return {"video": out, "model": self.model_id}


@register("kiwi_edit")
class KiwiEditBackend(Backend):
    """<video_edit> -- Kiwi-Edit-5B-Instruct, instruction-guided video editing.

    Replaces Text2Video-Zero (2023). Ships as a custom diffusers pipeline, so it
    is loaded with ``trust_remote_code=True``.
    """

    default_model_id = "linyq/kiwi-edit-5b-instruct-reference-diffusers"
    substitution = "Text2Video-Zero -> Kiwi-Edit-5B-Instruct"

    def load(self):
        from diffusers import DiffusionPipeline

        # Follow the reference implementation (showlab/Kiwi-Edit diffusers_demo.py):
        # load WITHOUT torch_dtype, then cast on .to(). Passing torch_dtype to
        # from_pretrained leaves some submodules in fp32 and inference dies with
        # "mat1 and mat2 must have the same dtype".
        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_id, trust_remote_code=True,
        )
        self.pipe.to(self.device, dtype=torch.bfloat16)
        try:
            self.pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

    def run(self, prompt, input_path, out_stem, step=None, **kw):
        if not input_path:
            raise ValueError("<video_edit> needs an input video")
        from ..media import read_frames

        max_frames = kw.get("max_frames", 81)
        frames = read_frames(input_path, max_frames=max_frames, size=None)
        # The DiT requires spatial dims to be multiples of 16.
        w, h = frames[0].size
        max_pixels = kw.get("max_pixels", 720 * 1280)
        scale = min(1.0, (max_pixels / float(w * h)) ** 0.5)
        w = max(16, int(w * scale) // 16 * 16)
        h = max(16, int(h * scale) // 16 * 16)
        frames = [f.resize((w, h)) for f in frames]
        # The temporal VAE compresses 4x, so the frame count must be 4n+1.
        n = max(1, ((len(frames) - 1) // 4) * 4 + 1)
        frames = frames[:n]

        video = self.pipe(
            prompt=prompt,
            source_video=frames,
            ref_image=kw.get("ref_image"),
            height=h, width=w,
            num_frames=len(frames),
            num_inference_steps=kw.get("steps", 50),
            guidance_scale=kw.get("guidance", 5.0),
            seed=kw.get("seed", 0),
            tiled=True,
        )
        if hasattr(video, "frames"):
            video = video.frames[0]
        out = f"{out_stem}.mp4"
        _save_video(video, out, fps=kw.get("fps", 15))
        return {"video": out, "model": self.model_id}


@register("depth_anything_v3")
class DepthAnything3Backend(Backend):
    """<image_depth> -- Depth Anything 3.

    NOTE: the DA3 checkpoints are released under CC-BY-NC-4.0 (non-commercial).
    Pass ``--backend-model image_depth=depth-anything/Depth-Anything-V2-Small-hf``
    (Apache-2.0, transformers-native) if you need a permissive licence.
    """

    default_model_id = "depth-anything/DA3MONO-LARGE"
    substitution = "Depth Anything V2 -> Depth Anything 3 (CC-BY-NC-4.0)"

    def load(self):
        self._mode = None
        # Preferred: the official depth-anything-3 package.
        try:
            from depth_anything_3.api import DepthAnything3

            self.model = DepthAnything3.from_pretrained(self.model_id).to(self.device)
            self.model.eval()
            self._mode = "da3"
            return
        except Exception as exc:
            print(f"[image_depth] depth-anything-3 unavailable ({exc}); "
                  f"falling back to Depth Anything V2")

        from transformers import pipeline as hf_pipeline

        self.pipe = hf_pipeline(
            "depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf",
            device=0 if self.device.startswith("cuda") else -1,
        )
        self._mode = "v2"
        self.substitution = "Depth Anything 3 package missing -> Depth Anything V2"

    def run(self, prompt, input_path, out_stem, step=None, **kw):
        if not input_path:
            raise ValueError("<image_depth> needs an input image")
        img = Image.open(input_path).convert("RGB")
        out = f"{out_stem}.png"

        if self._mode == "da3":
            import numpy as np

            with torch.no_grad():
                pred = self.model.inference([img])
            depth = pred.depth if hasattr(pred, "depth") else pred[0]
            if torch.is_tensor(depth):
                depth = depth.detach().float().cpu().numpy()
            depth = np.asarray(depth, dtype=np.float32)
            # DA3 returns (1, H, W) / (1, 1, H, W) depending on the checkpoint.
            depth = np.squeeze(depth)
            while depth.ndim > 2:
                depth = depth[0]
            lo, hi = float(depth.min()), float(depth.max())
            norm = (depth - lo) / max(hi - lo, 1e-8)
            Image.fromarray((norm * 255).astype("uint8"), mode="L").convert("RGB").save(out)
            return {"image": out, "model": self.model_id}

        self.pipe(img)["depth"].convert("RGB").save(out)
        return {"image": out, "model": "Depth-Anything-V2-Small-hf"}

"""Diffusion specialists: image generation/editing, controllable generation, video.

Backends here cover 20 of the 30 routing tokens and follow Table 9 of the paper:
Stable Diffusion XL (image generation), InstructPix2Pix (image editing),
ControlNet (controllable image generation), CogVideoX (video generation) and
Text2Video-Zero (video editing + controllable video generation).
"""

import os
from typing import Any, Dict, Optional

import torch
from PIL import Image

from .base import Backend, register, hf_kwargs

# ControlNet checkpoint per Olympus control condition (Figure 8).
CONTROLNET_IDS = {
    "canny": "lllyasviel/control_v11p_sd15_canny",
    "depth": "lllyasviel/control_v11f1p_sd15_depth",
    "normal": "lllyasviel/control_v11p_sd15_normalbae",
    "pose": "lllyasviel/control_v11p_sd15_openpose",
    "seg": "lllyasviel/control_v11p_sd15_seg",
    "scrib": "lllyasviel/control_v11p_sd15_scribble",
}
SD15_BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def _save_video(frames, path: str, fps: int = 8) -> str:
    import imageio
    import numpy as np

    arr = []
    for f in frames:
        if isinstance(f, Image.Image):
            f = np.array(f)
        if f.dtype != np.uint8:
            f = (np.clip(f, 0, 1) * 255).astype("uint8")
        arr.append(f)
    imageio.mimsave(path, arr, fps=fps, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p"])
    return path


@register("sdxl")
class SDXLBackend(Backend):
    """<image_gen> -- Stable Diffusion XL (Table 9)."""

    default_model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    def load(self):
        from diffusers import StableDiffusionXLPipeline

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_id, variant="fp16", use_safetensors=True, **hf_kwargs(self.dtype)
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    def run(self, prompt, input_path, out_stem, step=None, **kw):
        g = torch.Generator(device=self.device).manual_seed(kw.get("seed", 0))
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=kw.get("steps", 30),
            guidance_scale=kw.get("guidance", 5.0),
            generator=g,
        ).images[0]
        out = f"{out_stem}.png"
        image.save(out)
        return {"image": out}


@register("instructpix2pix")
class InstructPix2PixBackend(Backend):
    """<image_edit> -- InstructPix2Pix (Table 9)."""

    default_model_id = "timbrooks/instruct-pix2pix"

    def load(self):
        from diffusers import StableDiffusionInstructPix2PixPipeline

        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.model_id, safety_checker=None, **hf_kwargs(self.dtype)
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    def run(self, prompt, input_path, out_stem, step=None, **kw):
        if not input_path:
            raise ValueError("<image_edit> needs an input image "
                             "(chain it after <image_gen> or pass --input-image)")
        img = Image.open(input_path).convert("RGB")
        img.thumbnail((768, 768))
        g = torch.Generator(device=self.device).manual_seed(kw.get("seed", 0))
        out_img = self.pipe(
            prompt=prompt,
            image=img,
            num_inference_steps=kw.get("steps", 30),
            image_guidance_scale=kw.get("image_guidance", 1.5),
            guidance_scale=kw.get("guidance", 7.5),
            generator=g,
        ).images[0]
        out = f"{out_stem}.png"
        out_img.save(out)
        return {"image": out}


@register("controlnet")
class ControlNetBackend(Backend):
    """<{pose,canny,depth,normal,seg,scrib}_to_image> -- ControlNet (Table 9).

    The condition map is taken from the chained input when the previous step
    already produced one (e.g. ``<image_canny>`` -> ``<canny_to_image>``);
    otherwise it is derived from the raw input image with the matching detector.
    """

    default_model_id = SD15_BASE

    def load(self):
        # ControlNet weights depend on the step's condition, so the pipeline is
        # (re)built per condition inside run().
        self._pipes: Dict[str, Any] = {}

    def _pipe_for(self, condition: str):
        if condition in self._pipes:
            return self._pipes[condition]
        from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                               UniPCMultistepScheduler)

        cn = ControlNetModel.from_pretrained(CONTROLNET_IDS[condition],
                                             **hf_kwargs(self.dtype))
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.model_id, controlnet=cn, safety_checker=None, **hf_kwargs(self.dtype)
        ).to(self.device)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        self._pipes = {condition: pipe}  # keep only the active condition resident
        return pipe

    def run(self, prompt, input_path, out_stem, step=None, **kw):
        condition = (step.condition if step else None) or kw.get("condition", "canny")
        if not input_path:
            raise ValueError(f"<{condition}_to_image> needs an input image to "
                             f"derive the {condition} condition from")
        from ..conditions import ensure_condition_map

        cond_img = ensure_condition_map(input_path, condition,
                                        already_map=kw.get("already_map", False),
                                        device=self.device)
        cond_path = f"{out_stem}_cond.png"
        cond_img.save(cond_path)

        pipe = self._pipe_for(condition)
        g = torch.Generator(device=self.device).manual_seed(kw.get("seed", 0))
        img = pipe(
            prompt=prompt,
            image=cond_img,
            num_inference_steps=kw.get("steps", 30),
            guidance_scale=kw.get("guidance", 7.5),
            generator=g,
        ).images[0]
        out = f"{out_stem}.png"
        img.save(out)
        return {"image": out, "condition_map": cond_path}


@register("cogvideox")
class CogVideoXBackend(Backend):
    """<video_gen> -- CogVideoX (Table 9)."""

    default_model_id = "THUDM/CogVideoX-2b"

    def load(self):
        from diffusers import CogVideoXPipeline

        self.pipe = CogVideoXPipeline.from_pretrained(self.model_id,
                                                      **hf_kwargs(self.dtype))
        # Sequential offload keeps this within a single consumer GPU.
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_tiling()
        self.pipe.set_progress_bar_config(disable=True)

    def run(self, prompt, input_path, out_stem, step=None, **kw):
        g = torch.Generator().manual_seed(kw.get("seed", 0))
        frames = self.pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=kw.get("steps", 50),
            num_frames=kw.get("num_frames", 49),
            guidance_scale=kw.get("guidance", 6.0),
            generator=g,
        ).frames[0]
        out = f"{out_stem}.mp4"
        _save_video(frames, out, fps=kw.get("fps", 8))
        return {"video": out}


class _CrossFrameMixin:
    """Shared Text2Video-Zero setup: cross-frame attention over a frame batch."""

    @staticmethod
    def _apply_cross_frame(pipe):
        from diffusers.models.attention_processor import AttnProcessor2_0

        try:
            from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import (
                CrossFrameAttnProcessor,
            )
            pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
        except Exception:
            pipe.unet.set_attn_processor(AttnProcessor2_0())
        return pipe


@register("t2v_zero_edit")
class Text2VideoZeroEditBackend(Backend, _CrossFrameMixin):
    """<video_edit> -- Text2Video-Zero video editing (Table 9).

    Applies InstructPix2Pix frame-by-frame with cross-frame attention, which is
    the Text2Video-Zero recipe for temporally consistent video editing.
    """

    default_model_id = "timbrooks/instruct-pix2pix"

    def load(self):
        from diffusers import StableDiffusionInstructPix2PixPipeline

        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.model_id, safety_checker=None, **hf_kwargs(self.dtype)
        ).to(self.device)
        self._apply_cross_frame(self.pipe)
        self.pipe.set_progress_bar_config(disable=True)

    def run(self, prompt, input_path, out_stem, step=None, **kw):
        if not input_path:
            raise ValueError("<video_edit> needs an input video (or a chained image)")
        from ..media import read_frames

        frames = read_frames(input_path, max_frames=kw.get("max_frames", 16))
        g = torch.Generator(device=self.device).manual_seed(kw.get("seed", 0))
        edited = self.pipe(
            prompt=[prompt] * len(frames),
            image=frames,
            num_inference_steps=kw.get("steps", 20),
            image_guidance_scale=kw.get("image_guidance", 1.5),
            guidance_scale=kw.get("guidance", 7.5),
            generator=g,
        ).images
        out = f"{out_stem}.mp4"
        _save_video(edited, out, fps=kw.get("fps", 8))
        return {"video": out}


@register("t2v_zero_control")
class Text2VideoZeroControlBackend(Backend, _CrossFrameMixin):
    """<{cond}_to_video> -- Text2Video-Zero + ControlNet (Table 9)."""

    default_model_id = SD15_BASE

    def load(self):
        self._pipe = None
        self._condition = None

    def _pipe_for(self, condition: str):
        if self._pipe is not None and self._condition == condition:
            return self._pipe
        from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                               UniPCMultistepScheduler)

        cn = ControlNetModel.from_pretrained(CONTROLNET_IDS[condition],
                                             **hf_kwargs(self.dtype))
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.model_id, controlnet=cn, safety_checker=None, **hf_kwargs(self.dtype)
        ).to(self.device)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        self._apply_cross_frame(pipe)
        pipe.set_progress_bar_config(disable=True)
        self._pipe, self._condition = pipe, condition
        return pipe

    def run(self, prompt, input_path, out_stem, step=None, **kw):
        condition = (step.condition if step else None) or kw.get("condition", "canny")
        if not input_path:
            raise ValueError(f"<{condition}_to_video> needs an input image or video")
        from ..conditions import ensure_condition_map
        from ..media import read_frames

        n = kw.get("num_frames", 8)
        if input_path.lower().endswith((".mp4", ".avi", ".mov", ".webm", ".gif")):
            src = read_frames(input_path, max_frames=n)
        else:
            src = [Image.open(input_path).convert("RGB")] * n

        maps = [ensure_condition_map(f, condition,
                                     already_map=kw.get("already_map", False),
                                     device=self.device) for f in src]
        pipe = self._pipe_for(condition)
        g = torch.Generator(device=self.device).manual_seed(kw.get("seed", 0))
        frames = pipe(
            prompt=[prompt] * len(maps),
            image=maps,
            num_inference_steps=kw.get("steps", 20),
            guidance_scale=kw.get("guidance", 7.5),
            generator=g,
        ).images
        out = f"{out_stem}.mp4"
        _save_video(frames, out, fps=kw.get("fps", 4))
        return {"video": out}

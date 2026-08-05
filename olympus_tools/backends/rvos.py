"""Referring video object segmentation (``<video_ref_seg>``).

Table 9 names GLEE. GLEE needs a bespoke detectron2-based build, so the default
composes two specialists Olympus already depends on: GroundingDINO localises the
referring phrase in each frame and SAM converts that box into a mask. Both come
from ``transformers``, so this path needs no compilation.
"""

import json
from typing import List

import numpy as np
import torch
from PIL import Image

from .base import Backend, register, hf_kwargs


@register("rvos")
class RVOSBackend(Backend):
    """<video_ref_seg> -- referring video object segmentation."""

    default_model_id = "IDEA-Research/grounding-dino-base"

    def load(self):
        from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                                  SamModel, SamProcessor)

        self.det_processor = AutoProcessor.from_pretrained(self.model_id)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id).to(self.device).eval()
        sam_id = self.options.get("sam_model_id", "facebook/sam-vit-base")
        self.sam_processor = SamProcessor.from_pretrained(sam_id)
        self.sam = SamModel.from_pretrained(sam_id).to(self.device).eval()

    def _box_for(self, img: Image.Image, phrase: str):
        text = phrase.strip().lower()
        if not text.endswith("."):
            text += "."
        inputs = self.det_processor(images=img, text=text,
                                    return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.detector(**inputs)
        res = self.det_processor.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=0.25, text_threshold=0.2,
            target_sizes=[img.size[::-1]],
        )[0]
        if len(res["boxes"]) == 0:
            return None
        best = int(torch.argmax(res["scores"]))
        return [float(v) for v in res["boxes"][best]]

    def _mask_for(self, img: Image.Image, box) -> np.ndarray:
        inputs = self.sam_processor(img, input_boxes=[[box]],
                                    return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.sam(**inputs, multimask_output=False)
        masks = self.sam_processor.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]
        return masks[0][0].numpy().astype(bool)

    def run(self, prompt, input_path, out_stem, step=None, **kw):
        if not input_path:
            raise ValueError("<video_ref_seg> needs an input video")
        from ..media import read_frames, write_video

        frames = read_frames(input_path, max_frames=kw.get("max_frames", 16))
        overlays: List[Image.Image] = []
        hits = 0
        for frame in frames:
            box = self._box_for(frame, prompt)
            arr = np.array(frame)
            if box is not None:
                hits += 1
                mask = self._mask_for(frame, box)
                tint = arr.copy()
                tint[mask] = (0.45 * tint[mask] + 0.55 *
                              np.array([255, 40, 40])).astype(np.uint8)
                arr = tint
            overlays.append(Image.fromarray(arr))

        out = f"{out_stem}.mp4"
        write_video(overlays, out, fps=kw.get("fps", 8))
        meta = f"{out_stem}.json"
        with open(meta, "w") as fh:
            json.dump({"phrase": prompt, "frames": len(frames),
                       "frames_with_referent": hits}, fh, indent=2)
        return {"video": out, "annotation": meta,
                "frames_with_referent": f"{hits}/{len(frames)}"}

"""Perception specialists: detection, segmentation, grounding, and the
canny / depth / normal / pose estimators.

Table 9 assigns Co-DETR (detection), SegFormer (segmentation), GroundingDINO
(grounding), Depth Anything V2 (depth), Sapiens (normal), the OpenCV Canny
operator (canny) and DWPose (pose). SegFormer, GroundingDINO, Depth Anything V2,
Canny and DWPose are used exactly as specified. Co-DETR and Sapiens require
mmdetection / a bespoke build, so maintained drop-in equivalents are the default
and the paper's originals stay selectable via ``--backend-model``.
"""

import json
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageDraw

from .base import Backend, register, hf_kwargs

# Standard ADE20K 150-class palette (the palette ControlNet-seg was trained on).
_ADE = [
    [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3],
    [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230], [4, 250, 7],
    [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70], [8, 255, 51],
    [255, 6, 82], [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
    [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71],
    [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92], [112, 9, 255],
    [8, 255, 214], [7, 255, 224], [255, 184, 6], [10, 255, 71], [255, 41, 10],
    [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
    [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255],
    [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140], [250, 10, 15],
    [20, 255, 0], [31, 255, 0], [255, 31, 0], [255, 224, 0], [153, 255, 0],
    [0, 0, 255], [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
    [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255], [0, 255, 112],
    [0, 255, 133], [255, 0, 0], [255, 163, 0], [255, 102, 0], [194, 255, 0],
    [0, 143, 255], [51, 255, 0], [0, 82, 255], [0, 255, 41], [0, 255, 173],
    [10, 0, 255], [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
    [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20], [255, 184, 184],
    [0, 31, 255], [0, 255, 61], [0, 71, 255], [255, 0, 204], [0, 255, 194],
    [0, 255, 82], [0, 10, 255], [0, 112, 255], [51, 0, 255], [0, 194, 255],
    [0, 122, 255], [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
    [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0], [8, 184, 170],
    [133, 0, 255], [0, 255, 92], [184, 0, 255], [255, 0, 31], [0, 184, 255],
    [0, 214, 255], [255, 0, 112], [92, 255, 0], [0, 224, 255], [112, 224, 255],
    [70, 184, 160], [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
    [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0], [255, 0, 235],
    [245, 0, 255], [255, 0, 122], [255, 245, 0], [10, 190, 212], [214, 255, 0],
    [0, 204, 255], [20, 0, 255], [255, 255, 0], [0, 153, 255], [0, 41, 255],
    [0, 255, 204], [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
    [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255], [184, 255, 0],
    [0, 133, 255], [255, 214, 0], [25, 194, 194], [102, 255, 0], [92, 0, 255],
]


def ade_palette():
    return _ADE


def _draw_boxes(img: Image.Image, boxes, labels, scores=None) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x0, y0, x1, y1 = [float(v) for v in box]
        colour = tuple(_ADE[i % len(_ADE)])
        d.rectangle([x0, y0, x1, y1], outline=colour, width=3)
        tag = f"{label}" + (f" {scores[i]:.2f}" if scores is not None else "")
        d.text((x0 + 4, max(0, y0 - 12)), tag, fill=colour)
    return out


@register("canny")
class CannyBackend(Backend):
    """<image_canny> -- OpenCV Canny operator (Table 9, exact)."""

    def load(self):
        pass

    def run(self, prompt, input_path, out_stem, step=None, **kw):
        from ..conditions import canny_map

        if not input_path:
            raise ValueError("<image_canny> needs an input image")
        out = f"{out_stem}.png"
        canny_map(input_path, kw.get("low", 100), kw.get("high", 200)).save(out)
        return {"image": out}


@register("depth_anything")
class DepthAnythingBackend(Backend):
    """<image_depth> -- Depth Anything V2 (Table 9, exact)."""

    default_model_id = "depth-anything/Depth-Anything-V2-Small-hf"

    def load(self):
        from transformers import pipeline as hf_pipeline

        self.pipe = hf_pipeline("depth-estimation", model=self.model_id,
                                device=0 if self.device.startswith("cuda") else -1)

    def run(self, prompt, input_path, out_stem, step=None, **kw):
        if not input_path:
            raise ValueError("<image_depth> needs an input image")
        depth = self.pipe(Image.open(input_path).convert("RGB"))["depth"]
        out = f"{out_stem}.png"
        depth.convert("RGB").save(out)
        return {"image": out}


@register("normal")
class NormalBackend(Backend):
    """<image_normal> -- surface normal estimation.

    Table 9 names Sapiens; it needs a bespoke checkpoint + build, so the default
    is NormalBAE (the estimator ControlNet-normal was trained against, which also
    makes ``<image_normal>`` -> ``<normal_to_image>`` chain correctly).
    """


    def load(self):
        from ..conditions import _load

        self.model = _load("normal")

    def run(self, prompt, input_path, out_stem, step=None, **kw):
        if not input_path:
            raise ValueError("<image_normal> needs an input image")
        out = f"{out_stem}.png"
        self.model(Image.open(input_path).convert("RGB")).save(out)
        return {"image": out}


@register("dwpose")
class DWPoseBackend(Backend):
    """<image_pose> -- DWPose (Table 9); falls back to OpenPose if unavailable."""

    def load(self):
        self.model = None
        try:
            from controlnet_aux import DWposeDetector

            self.model = DWposeDetector()
            self._name = "DWPose"
        except Exception:
            from ..conditions import _load

            self.model = _load("pose")
            self._name = "OpenPose"
            self.model_id = "lllyasviel/Annotators"  # OpenPose weights

    def run(self, prompt, input_path, out_stem, step=None, **kw):
        if not input_path:
            raise ValueError("<image_pose> needs an input image")
        out = f"{out_stem}.png"
        self.model(Image.open(input_path).convert("RGB")).save(out)
        return {"image": out, "detector": self._name}


@register("segformer")
class SegFormerBackend(Backend):
    """<image_seg> -- SegFormer, ADE20K (Table 9, exact)."""

    default_model_id = "nvidia/segformer-b5-finetuned-ade-640-640"

    def load(self):
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        self.processor = SegformerImageProcessor.from_pretrained(self.model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.model_id).to(self.device).eval()

    def run(self, prompt, input_path, out_stem, step=None, **kw):
        if not input_path:
            raise ValueError("<image_seg> needs an input image")
        img = Image.open(input_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        seg = torch.nn.functional.interpolate(
            logits, size=img.size[::-1], mode="bilinear", align_corners=False
        ).argmax(dim=1)[0].cpu().numpy()

        palette = np.array(_ADE, dtype=np.uint8)
        colour = np.zeros((*seg.shape, 3), dtype=np.uint8)
        for label in np.unique(seg):
            colour[seg == label] = palette[label % len(palette)]
        mask_path, overlay_path = f"{out_stem}.png", f"{out_stem}_overlay.png"
        Image.fromarray(colour).save(mask_path)
        Image.blend(img, Image.fromarray(colour), 0.5).save(overlay_path)

        id2label = self.model.config.id2label
        present = [id2label.get(int(i), str(i)) for i in np.unique(seg)]
        meta = f"{out_stem}.json"
        with open(meta, "w") as fh:
            json.dump({"classes": present}, fh, indent=2)
        return {"image": mask_path, "overlay": overlay_path,
                "annotation": meta, "classes": present}


@register("detection")
class DetectionBackend(Backend):
    """<image_det> -- object detection.

    Table 9 names Co-DETR, which requires an mmdetection build; DETR is the
    default drop-in. Pass ``--backend-model detection=<hf-id>`` to swap in any
    HF detection checkpoint.
    """

    default_model_id = "facebook/detr-resnet-101"

    def load(self):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForObjectDetection.from_pretrained(
            self.model_id).to(self.device).eval()

    def run(self, prompt, input_path, out_stem, step=None, **kw):
        if not input_path:
            raise ValueError("<image_det> needs an input image")
        img = Image.open(input_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        res = self.processor.post_process_object_detection(
            outputs, threshold=kw.get("threshold", 0.7),
            target_sizes=torch.tensor([img.size[::-1]]).to(self.device),
        )[0]
        labels = [self.model.config.id2label[int(i)] for i in res["labels"]]
        boxes = res["boxes"].tolist()
        scores = res["scores"].tolist()
        vis = f"{out_stem}.png"
        _draw_boxes(img, boxes, labels, scores).save(vis)
        meta = f"{out_stem}.json"
        with open(meta, "w") as fh:
            json.dump([{"label": l, "box": b, "score": s}
                       for l, b, s in zip(labels, boxes, scores)], fh, indent=2)
        return {"image": vis, "annotation": meta, "detections": labels}


@register("grounding_dino")
class GroundingDINOBackend(Backend):
    """<image_ground> -- GroundingDINO (Table 9, exact).

    The routing-token payload is the referring phrase to localise.
    """

    default_model_id = "IDEA-Research/grounding-dino-base"

    def load(self):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id).to(self.device).eval()

    def run(self, prompt, input_path, out_stem, step=None, **kw):
        if not input_path:
            raise ValueError("<image_ground> needs an input image")
        img = Image.open(input_path).convert("RGB")
        # GroundingDINO expects lowercase phrases terminated by a period.
        text = prompt.strip().lower()
        if not text.endswith("."):
            text += "."
        inputs = self.processor(images=img, text=text,
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=kw.get("threshold", 0.3),
            text_threshold=kw.get("text_threshold", 0.25),
            target_sizes=[img.size[::-1]],
        )[0]
        labels = [str(l) for l in res["labels"]]
        boxes = res["boxes"].tolist()
        scores = res["scores"].tolist()
        vis = f"{out_stem}.png"
        _draw_boxes(img, boxes, labels, scores).save(vis)
        meta = f"{out_stem}.json"
        with open(meta, "w") as fh:
            json.dump({"phrase": prompt,
                       "boxes": [{"label": l, "box": b, "score": s}
                                 for l, b, s in zip(labels, boxes, scores)]},
                      fh, indent=2)
        return {"image": vis, "annotation": meta, "boxes": boxes}

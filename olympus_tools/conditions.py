"""Condition-map extraction for controllable generation.

The six Olympus control conditions (pose, canny, depth, normal, seg, scrib) each
need a condition map. When a ``<{cond}_to_image>`` token is chained directly
after the matching estimation token (e.g. ``<image_canny>`` -> ``<canny_to_image>``)
the upstream artifact *is* the map and is used as-is. Otherwise the map is
derived here from the raw image, so a user can hand Olympus an ordinary photo and
still get controllable generation.
"""

from typing import Union

import numpy as np
from PIL import Image

_DETECTORS = {}


def _load(name: str):
    """Lazily build a controlnet_aux detector, cached by name."""
    if name in _DETECTORS:
        return _DETECTORS[name]
    from controlnet_aux import (HEDdetector, NormalBaeDetector, OpenposeDetector,
                                PidiNetDetector)

    if name == "pose":
        det = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    elif name == "normal":
        det = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
    elif name == "scrib":
        det = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
    elif name == "hed":
        det = HEDdetector.from_pretrained("lllyasviel/Annotators")
    else:
        raise KeyError(name)
    _DETECTORS[name] = det
    return det


def _as_image(src: Union[str, Image.Image]) -> Image.Image:
    if isinstance(src, Image.Image):
        return src.convert("RGB")
    return Image.open(src).convert("RGB")


def canny_map(src, low: int = 100, high: int = 200) -> Image.Image:
    """OpenCV Canny operator -- the estimator named in Table 9."""
    import cv2

    img = np.array(_as_image(src))
    edges = cv2.Canny(img, low, high)
    return Image.fromarray(np.stack([edges] * 3, axis=-1))


def depth_map(src, device: str = "cuda") -> Image.Image:
    """Depth Anything V2 (Table 9)."""
    from transformers import pipeline as hf_pipeline

    global _DETECTORS
    if "depth" not in _DETECTORS:
        _DETECTORS["depth"] = hf_pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=0 if device.startswith("cuda") else -1,
        )
    out = _DETECTORS["depth"](_as_image(src))["depth"]
    return out.convert("RGB")


def seg_map(src, device: str = "cuda") -> Image.Image:
    """SegFormer ADE20K semantic map, colourised (Table 9)."""
    from .backends.perception import ade_palette
    import torch
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    global _DETECTORS
    if "seg" not in _DETECTORS:
        mid = "nvidia/segformer-b5-finetuned-ade-640-640"
        _DETECTORS["seg"] = (
            SegformerImageProcessor.from_pretrained(mid),
            SegformerForSemanticSegmentation.from_pretrained(mid).to(device).eval(),
        )
    proc, model = _DETECTORS["seg"]
    img = _as_image(src)
    inputs = proc(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    seg = torch.nn.functional.interpolate(
        logits, size=img.size[::-1], mode="bilinear", align_corners=False
    ).argmax(dim=1)[0].cpu().numpy()
    palette = np.array(ade_palette(), dtype=np.uint8)
    colour = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for label in np.unique(seg):
        colour[seg == label] = palette[label % len(palette)]
    return Image.fromarray(colour)


def ensure_condition_map(src, condition: str, already_map: bool = False,
                         device: str = "cuda") -> Image.Image:
    """Return the condition map for ``condition``.

    ``already_map=True`` means the caller knows ``src`` is a condition map from an
    upstream estimation step, so it is passed through untouched.
    """
    img = _as_image(src)
    if already_map:
        return img
    if condition == "canny":
        return canny_map(img)
    if condition == "depth":
        return depth_map(img, device=device)
    if condition == "seg":
        return seg_map(img, device=device)
    if condition in ("pose", "normal", "scrib"):
        return _load(condition)(img)
    raise ValueError(f"unknown control condition: {condition}")

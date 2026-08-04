"""Small media helpers shared by the video backends."""

import os
from typing import List

from PIL import Image


def read_frames(path: str, max_frames: int = 16, size=(512, 512)) -> List[Image.Image]:
    """Read up to ``max_frames`` evenly spaced RGB frames from a video or GIF.

    ``size=None`` keeps the source resolution. A still image is accepted too and
    repeated, which lets ``<video_edit>`` and the ``<{cond}_to_video>`` tokens
    chain off an upstream image step.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        img = Image.open(path).convert("RGB")
        if size is not None:
            img = img.resize(size)
        return [img] * max_frames

    import imageio.v3 as iio

    frames = []
    for i, frame in enumerate(iio.imiter(path)):
        frames.append(Image.fromarray(frame).convert("RGB"))
        if i > 512:  # guard against absurdly long inputs
            break
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    if len(frames) > max_frames:
        step = len(frames) / max_frames
        frames = [frames[int(i * step)] for i in range(max_frames)]
    return frames if size is None else [f.resize(size) for f in frames]


def write_video(frames, path: str, fps: int = 8) -> str:
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

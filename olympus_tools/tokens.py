"""Task-specific routing tokens for Olympus.

This module is the single source of truth mapping every routing token emitted by
the Olympus router to (a) the task it denotes, (b) the specialist backend that
executes it, and (c) its dataflow signature -- what artifact it consumes and what
it produces.

The token set follows Figure 8 of the paper (20 tasks; 18 base tokens plus six
controllable-image and six controllable-video condition tokens = 30 tokens), and
the ``paper_model`` field records the specialist named in Table 9.

Note on ``<image_denoise>``: Figure 8 prints this as ``<image_denosie>``, but the
released OlympusInstruct data (``image_denoise.json``) uses ``<image_denoise>``,
which is what the model was actually trained to emit. We accept both.
"""

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

# Artifact kinds flowing between chained steps.
NONE = "none"
IMAGE = "image"
VIDEO = "video"
MESH = "mesh"
ANNOTATION = "annotation"


@dataclass(frozen=True)
class TaskSpec:
    """Static description of one routing token."""

    token: str  # bare token name, e.g. "image_gen" for <image_gen>
    task: str  # human-readable task name
    backend: str  # backend id resolved through olympus_tools.registry
    paper_model: str  # specialist named in Table 9 of the paper
    consumes: str = NONE  # artifact this task needs as input
    produces: str = IMAGE  # artifact this task emits
    aliases: tuple = field(default=())  # alternate spellings seen in the wild
    condition: Optional[str] = None  # control condition for CIG/CVG tokens

    @property
    def open_tag(self) -> str:
        return f"<{self.token}>"

    @property
    def close_tag(self) -> str:
        return f"</{self.token}>"


# The six control conditions used by controllable image/video generation.
CONTROL_CONDITIONS = ("pose", "canny", "depth", "normal", "seg", "scrib")

_BASE_TASKS: List[TaskSpec] = [
    # ---- image generation / editing -------------------------------------
    TaskSpec("image_gen", "Image Generation", "qwen_image", "Stable Diffusion XL",
             consumes=NONE, produces=IMAGE),
    TaskSpec("image_edit", "Image Editing", "qwen_image_edit", "InstructPix2Pix",
             consumes=IMAGE, produces=IMAGE),
    # ---- image restoration ----------------------------------------------
    TaskSpec("image_deblur", "Image Deblurring", "instructir", "InstructIR",
             consumes=IMAGE, produces=IMAGE),
    TaskSpec("image_denoise", "Image Denoising", "instructir", "InstructIR",
             consumes=IMAGE, produces=IMAGE, aliases=("image_denosie",)),
    TaskSpec("image_derain", "Image Deraining", "instructir", "InstructIR",
             consumes=IMAGE, produces=IMAGE),
    TaskSpec("image_sr", "Image Super-Resolution", "swin2sr", "Swin2SR",
             consumes=IMAGE, produces=IMAGE),
    # ---- image perception ------------------------------------------------
    TaskSpec("image_det", "Object Detection", "detection", "Co-DETR",
             consumes=IMAGE, produces=ANNOTATION),
    TaskSpec("image_seg", "Object Segmentation", "segformer", "SegFormer",
             consumes=IMAGE, produces=ANNOTATION),
    TaskSpec("image_ground", "Visual Grounding", "grounding_dino", "GroundingDINO",
             consumes=IMAGE, produces=ANNOTATION),
    TaskSpec("image_depth", "Depth Estimation", "depth_anything_v3", "Depth Anything V2",
             consumes=IMAGE, produces=IMAGE),
    TaskSpec("image_normal", "Normal Estimation", "normal", "Sapiens",
             consumes=IMAGE, produces=IMAGE),
    TaskSpec("image_canny", "Canny Estimation", "canny", "OpenCV Canny Operator",
             consumes=IMAGE, produces=IMAGE),
    TaskSpec("image_pose", "Pose Estimation", "dwpose", "DWPose",
             consumes=IMAGE, produces=IMAGE),
    # ---- video ------------------------------------------------------------
    TaskSpec("video_gen", "Video Generation", "wan_video", "CogVideoX",
             consumes=NONE, produces=VIDEO),
    TaskSpec("video_edit", "Video Editing", "kiwi_edit", "Text2Video-Zero",
             consumes=VIDEO, produces=VIDEO),
    TaskSpec("video_ref_seg", "Referring Video Object Segmentation", "rvos", "GLEE",
             consumes=VIDEO, produces=VIDEO),
    # ---- 3D ---------------------------------------------------------------
    TaskSpec("3D_gen_text", "Text-to-3D Generation", "text_to_3d", "LGM",
             consumes=NONE, produces=MESH),
    TaskSpec("3D_gen_image", "Image-to-3D Generation", "image_to_3d", "Wonder3D",
             consumes=IMAGE, produces=MESH),
]


def _controllable_tasks() -> List[TaskSpec]:
    """Build the 12 controllable image/video generation tokens."""
    out: List[TaskSpec] = []
    for cond in CONTROL_CONDITIONS:
        out.append(TaskSpec(
            f"{cond}_to_image", f"Controllable Image Generation ({cond})",
            "controlnet", "ControlNet",
            consumes=IMAGE, produces=IMAGE, condition=cond,
        ))
        out.append(TaskSpec(
            f"{cond}_to_video", f"Controllable Video Generation ({cond})",
            "t2v_zero_control", "Text2Video-Zero",
            consumes=IMAGE, produces=VIDEO, condition=cond,
        ))
    return out


ALL_TASKS: List[TaskSpec] = _BASE_TASKS + _controllable_tasks()

# Backends named in Table 9 of the paper. The defaults above point at current
# state-of-the-art models instead; pass ``--legacy-backends`` to run_tools.py to
# reproduce the paper's exact configuration.
LEGACY_BACKENDS: Dict[str, str] = {
    "image_gen": "sdxl",                 # Stable Diffusion XL
    "image_edit": "instructpix2pix",     # InstructPix2Pix
    "image_depth": "depth_anything",     # Depth Anything V2
    "video_gen": "cogvideox",            # CogVideoX
    "video_edit": "t2v_zero_edit",       # Text2Video-Zero
}


def apply_legacy_backends() -> None:
    """Switch the affected tokens back to their Table 9 specialists in place."""
    global ALL_TASKS, TOKEN_TO_SPEC
    swapped = []
    for spec in ALL_TASKS:
        legacy = LEGACY_BACKENDS.get(spec.token)
        swapped.append(replace(spec, backend=legacy) if legacy else spec)
    ALL_TASKS = swapped
    TOKEN_TO_SPEC = {}
    for spec in ALL_TASKS:
        TOKEN_TO_SPEC[spec.token] = spec
        for alias in spec.aliases:
            TOKEN_TO_SPEC[alias] = spec


# token (and alias) -> TaskSpec
TOKEN_TO_SPEC: Dict[str, TaskSpec] = {}
for _spec in ALL_TASKS:
    TOKEN_TO_SPEC[_spec.token] = _spec
    for _alias in _spec.aliases:
        TOKEN_TO_SPEC[_alias] = _spec


def get_spec(token: str) -> Optional[TaskSpec]:
    """Look up a token, tolerating case differences (e.g. ``3d_gen_text``)."""
    if token in TOKEN_TO_SPEC:
        return TOKEN_TO_SPEC[token]
    lowered = {k.lower(): v for k, v in TOKEN_TO_SPEC.items()}
    return lowered.get(token.lower())


def known_tokens() -> List[str]:
    return [s.token for s in ALL_TASKS]

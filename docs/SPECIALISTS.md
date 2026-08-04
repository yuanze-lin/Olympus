# Specialist models

All 30 routing tokens from Figure 8 are dispatched. Several Table 9 entries date
from 2023-2024, so the defaults point at current open-weight models instead.
`--legacy-backends` restores the paper's configuration. Which model produced each
artifact is recorded per step in `manifest.json`.

| Routing token | Task | Default | Paper (Table 9) |
|---|---|---|---|
| `<image_gen>` | Image Generation | **Qwen-Image** | Stable Diffusion XL |
| `<image_edit>` | Image Editing | **Qwen-Image-Edit-2509** | InstructPix2Pix |
| `<video_gen>` | Video Generation | **Wan2.2-TI2V-5B** | CogVideoX |
| `<video_edit>` | Video Editing | **Kiwi-Edit-5B-Instruct** | Text2Video-Zero |
| `<image_depth>` | Depth Estimation | **Depth Anything 3** | Depth Anything V2 |
| `<image_deblur>` `<image_denoise>` `<image_derain>` | Restoration | InstructIR | InstructIR |
| `<image_sr>` | Super-Resolution | Swin2SR | Swin2SR |
| `<image_seg>` | Object Segmentation | SegFormer | SegFormer |
| `<image_ground>` | Visual Grounding | GroundingDINO | GroundingDINO |
| `<image_canny>` | Canny Estimation | OpenCV Canny | OpenCV Canny |
| `<image_pose>` | Pose Estimation | DWPose | DWPose |
| `<image_det>` | Object Detection | DETR | Co-DETR *(needs mmdetection)* |
| `<image_normal>` | Normal Estimation | NormalBAE | Sapiens *(bespoke build)* |
| `<video_ref_seg>` | Referring Video Seg. | GroundingDINO + SAM | GLEE *(needs detectron2)* |
| `<3D_gen_text>` | Text-to-3D | **TRELLIS-text-base** | LGM |
| `<3D_gen_image>` | Image-to-3D | **TRELLIS.2-4B** | Wonder3D |
| `<{pose,canny,depth,normal,seg,scrib}_to_image>` | Controllable Image Gen | ControlNet (per condition) | ControlNet |
| `<{pose,canny,depth,normal,seg,scrib}_to_video>` | Controllable Video Gen | Text2Video-Zero + ControlNet | Text2Video-Zero |

## 3D fallbacks

A 3D token whose backend is not installed downgrades instead of failing:
`<3D_gen_image>` tries TRELLIS.2-4B, then Hunyuan3D-2, then Shap-E;
`<3D_gen_text>` tries TRELLIS-text-base, then Shap-E. The chosen backend is
printed at startup and recorded in `manifest.json` as `substitution`.

## Licensing

Depth Anything 3 is CC-BY-NC-4.0 (non-commercial); for a permissive alternative
use `--backend-model image_depth=depth-anything/Depth-Anything-V2-Small-hf`
(Apache-2.0). Hunyuan3D-2 is under `tencent-hunyuan-community`, not MIT/Apache.

## Behaviour

* One GPU is enough. Specialists load lazily and each is evicted before the next
  loads, so peak VRAM is roughly one model. Large models use CPU offload.
* Condition maps are derived automatically. `<canny_to_image>` on a raw photo
  computes the edge map first; chained after `<image_canny>` it reuses the
  upstream map.
* A specialist that errors is recorded in `manifest.json` and the remaining steps
  still run.
* To add a backend, subclass `Backend`, decorate with `@register("my_backend")`,
  and point a token at it in `olympus_tools/tokens.py`.

## Dependency pins

`requirements_tools.txt` requires `transformers >= 4.50, < 5.0` (verified on
4.57.6). Older releases break Qwen-Image's text encoder; 5.x drops an API the
router's vision tower needs.

On `transformers >= 4.50` the router used to fail silently, producing fluent text
but no routing tokens. That is fixed. If you change a pin, re-check with
`python run_tools.py --prompt "..." --dry-run`: an empty plan means the router
regressed.

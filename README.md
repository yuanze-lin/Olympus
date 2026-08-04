<p align="center"><img src="https://github.com/yuanze-lin/Olympus/blob/main/asset/olympus.png" alt="icon" width="150" height="150" style="vertical-align:middle; margin-right:5px;" /></p>

# Olympus: A Universal Task Router for Computer Vision Tasks (CVPR 2025, Highlight) <br/>

[![PDF](https://img.shields.io/badge/PDF-Download-orange?style=flat-square&logo=adobeacrobatreader&logoColor=white)](https://arxiv.org/pdf/2412.09612)
[![arXiv](https://img.shields.io/badge/arXiv-2412.09612-b31b1b.svg)](https://arxiv.org/abs/2412.09612)
[![Project Page](https://img.shields.io/badge/Project%20Page-Visit%20Now-0078D4?style=flat-square&logo=googlechrome&logoColor=white)](https://yuanze-lin.me/Olympus_page/)
[![Weights](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E)](https://huggingface.co/Yuanze/Olympus)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-FFD21E)](https://huggingface.co/datasets/Yuanze/Olympus)
[![YouTube Video](https://img.shields.io/badge/YouTube%20Video-FF0000?style=flat-square&logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=N1xOdIrVvn4)

Official implementation of "Olympus: A Universal Task Router for Computer Vision Tasks" 

[Yuanze Lin](https://yuanze-lin.me/), [Yunsheng Li](https://scholar.google.com/citations?user=hJrIyCwAAAAJ&hl=en), [Dongdong Chen](https://www.dongdongchen.bid/), [Weijian Xu](https://weijianxu.com/), [Ronald Clark](https://www.ron-clark.com/), [Philip H. S. Torr](https://eng.ox.ac.uk/people/philip-torr/)

**:hearts: If you find our project is helpful for your research, please kindly give us a :star2: and cite our paper :bookmark_tabs:   : )**

## :mega:  News
- [x] Release the code for integration with task-specific models.
- [x] Release the training & inference code.
- [x] Release Olympus datasets.
- [x] Release the model of Olympus.


## :low_brightness: Overview 

![image](https://github.com/yuanze-lin/Olympus/blob/main/asset/overview.png)

  
## Getting Started

### :hammer_and_wrench: Environment Installation <a href="#install" id="install"/>
To establish the environment, just run this code in the shell:
```
git clone https://github.com/yuanze-lin/Olympus.git
cd Olympus
conda create -n olympus python==3.10 -y
conda activate olympus
pip install -r requirements.txt
```
That will create the environment ```olympus``` we used.

That is all you need to run the router on its own. To also **execute** the routed
tasks and generate real images, videos and 3D models with
[`run_tools.py`](#specialists), install the specialist stack into the *same*
environment:
```
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements_tools.txt
bash scripts/install_specialists.sh
```
The router and every specialist share this one environment; there is no
per-tool environment to manage.

**3D backends.** `<3D_gen_image>` uses
[TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) and `<3D_gen_text>`
uses [TRELLIS-text-base](https://huggingface.co/microsoft/TRELLIS-text-base), both
producing textured meshes with PBR materials. They compile several CUDA
extensions, so they install separately:
```
bash scripts/install_3d.sh
```
That builds both, plus Hunyuan3D-2 as an ungated fallback for `<3D_gen_image>`.
Each step is independent, so a failure in one does not block the others, and any
3D token whose backend is missing falls back automatically rather than erroring.
See the [3D fallback chain](#supported-tasks-and-specialist-models) below.

> **TRELLIS.2-4B needs two gated Hugging Face repos at runtime:**
> [`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
> (image conditioning) and [`briaai/RMBG-2.0`](https://huggingface.co/briaai/RMBG-2.0)
> (background removal). Accept both licenses on the Hub while logged in
> (`huggingface-cli login`) and nothing else is needed.
>
> If your request is rejected, obtain the weights locally (see
> [microsoft/TRELLIS.2#38](https://github.com/microsoft/TRELLIS.2/issues/38)) and
> point at the folders:
> ```
> DINO_MODEL_PATH=/path/to/dinov3-vitl16-pretrain-lvd1689m \
> SEG_MODEL_PATH=/path/to/RMBG-2.0 \
>   python run_tools.py --prompt "..." --input-image assets/room.jpg
> ```
> The Hub call is tried first; these apply only if it fails. Or skip TRELLIS.2 and
> let `<3D_gen_image>` fall back to Hunyuan3D-2, which is ungated.

### Download Models & Data ###
We share our collected Olympus dataset as follows:

| Instruction    | Link |
|---------|------|
| Olympus Dataset | [Olympus_dataset](https://huggingface.co/datasets/Yuanze/Olympus) |
| Olympus Fine-tuning Data | [Olympus.json](https://huggingface.co/datasets/Yuanze/Olympus/blob/main/Olympus.json) |

- ```Olympus_dataset```: There are 20 JSON files under ```20 individual tasks``` folder, each corresponding to a specific task. You can refer to the routing token definitions in our paper to identify the task associated with each JSON file, along with the chain-of-action data provided in ```coa.json```. Each of these 21 JSON files includes both training and test data. ```OlympusInstruct.json``` and ```OlympusBench.json``` contain the collected OlympusInstruct and OlympusBench datasets, respectively.
- ```Olympus.json```: The final instruction data for fine-tuning.


(1) Download the Olympus model:
```
python download_olympus.py
```
It will save the ```Olympus``` model under the ```ckpts``` folder.

(2) Download the Olympus data for fine-tuning:
```
python download_olympus_dataset.py
```
It saves the fine-tuning instruction data ```Olympus.json``` to the ```train_data``` folder, while all other JSON files are stored in the newly created ```jsons``` folder. Note that ```Olympus.json``` is a combination of ```llava_v1_5_mix665k.json``` and OlympusInstruct, our collected instruction data covering 20 tasks.

**If you want to merge the data manually, download [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json) into the ```jsons``` folder, then run the merge script:**

```
python scripts/merge_data.py
```
You can specify which tasks to merge by referring to the script ```scripts/merge_tasks.py```.

(3) Download the Mipha-3B model for fine-tuning:
```
python download_mipha_3b.py
```
It will save the ```Mipha-3B``` model under the ```ckpts``` folder.

### :rocket: Inference <a href="#specialists" id="specialists"/>

Give Olympus one instruction and get the finished assets back. It routes the
instruction to the right specialists, **calls them**, and chains them together, so
you end up with real images, videos and 3D meshes rather than routing tokens
([#1](https://github.com/yuanze-lin/Olympus/issues/1)).

```
                    ┌──────────────┐   routing tokens   ┌───────────────────┐
 user instruction → │   Olympus    │ ─────────────────▶ │  olympus_tools    │ → PNG / MP4 / GLB
                    │  (router)    │  <image_gen>...    │   30 tokens       │
                    └──────────────┘                    └───────────────────┘
```

#### Model weights

Requires the specialist stack from [Environment Installation](#install).
Weights stream from the Hugging Face Hub on first use. The defaults are large
(~200 GB if you exercise every token), so point the cache at a big disk:

```
export HF_HOME=/path/to/big/disk/hf_cache
python scripts/prefetch_specialists.py     # optional: pre-download everything
```

#### Quick start

One instruction in, four finished assets out:

```
python run_tools.py \
  --prompt "Generate an image of a fluffy orange cat lounging on a windowsill, \
with sunlight streaming through the glass and casting soft shadows to create a cozy atmosphere. \
Next, would it be possible to change the cat's color to white? This change will make it more eye-catching. \
In the following step, produce a high-resolution 3D model based on the modified image. \
At the next point, please show a video of a cat and a dog running on a playground." \
  --model-path ckpts/Olympus \
  --output-dir outputs/cat
```

Olympus routes the instruction to four specialists and **chains them
automatically**: the edit runs on the generated image, and the 3D model is built
from the edited image:

```
Execution plan:
  [0] <image_gen> via qwen_image
        "a fluffy orange cat lounging on a windowsill, ..."
  [1] <image_edit> via qwen_image_edit  <- step 0
        "change the cat's color to white."
  [2] <3D_gen_image> via trellis2  <- step 1
        "produce a high-resolution 3D model based on the modified image."
  [3] <video_gen> via wan_video
        "a cat and a dog running on a playground."
```

```
outputs/cat/
├── plan.json                       # parsed routing tokens + dataflow
├── manifest.json                   # what ran, how long, which model, where it landed
├── step0_image_gen.png
├── step1_image_edit.png
├── step2_3D_gen_image.glb          # textured mesh with PBR materials
└── step3_video_gen.mp4
```

Measured on a single 48 GB GPU (the four steps above, full quality):

| Step | Model | Time |
|---|---|---|
| `<image_gen>` | Qwen-Image | 131.5 s |
| `<image_edit>` | Qwen-Image-Edit-2509 | 187.8 s |
| `<3D_gen_image>` | Hunyuan3D-2 | 280.1 s |
| `<video_gen>` | Wan2.2-TI2V-5B | 191.5 s |

The `<3D_gen_image>` row is the Hunyuan3D-2 fallback: the measurement host had no
access to TRELLIS.2's gated weights, so the default backend downgraded (see the
[3D fallback chain](#supported-tasks-and-specialist-models)). Expect TRELLIS.2-4B
to be slower and considerably higher quality.

Useful flags:

| Flag | Purpose |
|---|---|
| `--dry-run` | print the plan without loading any specialist (no GPU needed) |
| `--list-tokens` | list all 30 routing tokens with their backends |
| `--input-image path.png` | seed image for tasks that edit/analyse an image |
| `--router-output "<image_gen>...</image_gen>"` | skip the router, run tokens directly |
| `--plan outputs/cat/plan.json` | re-run a saved plan |
| `--legacy-backends` | use the paper's Table 9 specialists instead of the SOTA defaults |
| `--backend-model image_gen=<hf-id>` | swap any single checkpoint |
| `--step-option video_gen.num_frames=25` | per-token knobs |
| `--max-resident 2` | keep more than one specialist in VRAM |

Tasks that need an image work the same way:

```
python run_tools.py --prompt "Segment everything in this photo, then estimate its depth map." \
  --input-image assets/room.jpg --output-dir outputs/room
```

#### Supported tasks and specialist models

All **30 routing tokens** from Figure 8 are dispatched. Because several models in
Table 9 date from 2023–2024, the defaults now point at current open-weight
state-of-the-art models; `--legacy-backends` restores the paper's configuration.

| Routing token | Task | Default (SOTA) | Paper (Table 9) |
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
| `<3D_gen_text>` | Text-to-3D | **TRELLIS-text-base** → Shap-E | LGM |
| `<3D_gen_image>` | Image-to-3D | **TRELLIS.2-4B** → Hunyuan3D-2 → Shap-E | Wonder3D |
| `<{pose,canny,depth,normal,seg,scrib}_to_image>` | Controllable Image Gen | ControlNet (per condition) | ControlNet |
| `<{pose,canny,depth,normal,seg,scrib}_to_video>` | Controllable Video Gen | Text2Video-Zero + ControlNet | Text2Video-Zero |

Which model produced each artifact is recorded per step in `manifest.json`.

> **Licensing note.** Depth Anything 3 checkpoints are released under
> CC-BY-NC-4.0 (non-commercial). For a permissive alternative use
> `--backend-model image_depth=depth-anything/Depth-Anything-V2-Small-hf`
> (Apache-2.0). Hunyuan3D-2 (the automatic fallback for `<3D_gen_image>`) is
> released under the `tencent-hunyuan-community` license, not MIT/Apache.
> Check its terms before commercial use.

#### Notes

- **3D tokens degrade automatically, in order.** `<3D_gen_image>` tries
  TRELLIS.2-4B first; if its gated DINOv3/RMBG weights aren't reachable (no Hub
  access and no `DINO_MODEL_PATH`/`SEG_MODEL_PATH` override, see above) it prints
  why and drops to Hunyuan3D-2, then to Shap-E if that isn't installed either.
  `<3D_gen_text>` tries TRELLIS-text-base, then Shap-E. Every run still produces
  a mesh; `manifest.json`'s `substitution` field records which model actually
  ran, so nothing is silently swapped without a paper trail.

- **One GPU is enough.** Specialists load lazily and the previous one is evicted
  before the next is built, so peak VRAM is roughly a single model rather than the
  sum. The router is freed before the first specialist loads. The large models
  (Qwen-Image, Wan2.2) use CPU offload; on a 48 GB card `<image_gen>` takes about
  6 minutes at full quality; lower `--step-option image_gen.steps=15` to trade
  quality for speed.
- **Condition maps are handled for you.** `<canny_to_image>` on a raw photo derives
  the edge map first; chained after `<image_canny>` the upstream map is reused
  instead of being recomputed.
- **Failures are isolated.** A specialist that errors is recorded in
  `manifest.json` and the remaining steps still run.
- **Extending it** takes one class: subclass `Backend`, decorate with
  `@register("my_backend")`, and point a token at it in `olympus_tools/tokens.py`.
- **Just want the routing tokens?** `--dry-run` prints the plan without touching a
  GPU. `predict.py` / `predict.sh` are also still there and unchanged, if you want
  the raw router output as text for analysis rather than finished assets.

#### Before you upgrade dependencies

`requirements_tools.txt` pins a deliberately narrow window: `transformers` must
be **>= 4.50 and < 5.0** (verified on 4.57.6). Older releases are too old for
Qwen-Image's text encoder, and 5.x drops an API the router's vision tower needs;
the details are documented inline in that file.

Note that on `transformers >= 4.50` the router previously failed *silently*: it
still produced fluent text but emitted no routing tokens at all. That is fixed
here, but if you change any pin, always re-check that the router still emits
routing tokens: `python run_tools.py --prompt "..." --dry-run` prints the parsed
plan without loading a single specialist, so an empty plan means the router
regressed.

Verify an install with:

```
python run_tools.py --list-tokens          # all 30 tokens and their backends
python scripts/smoke_test_tokens.py --fast # runs every token, writes smoke_results.json
```


### Visual Instruction Tuning
Please refer [here](https://github.com/haotian-liu/LLaVA/blob/9a26bd1435b4ac42c282757f2c16d34226575e96/README.md#visual-instruction-tuning) to prepare the instruction tuning data. Especially, store the images from different datasets under ```train_data``` folder.

Run the following code to fine-tune the model: 
```
bash scripts/mipha/finetune.sh
```

### Evaluation
To evaluate the model's performance on different benchmarks:

See [Evaluation.md](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).

Please place the evaluation data under the ```eval``` folder. The evaluation scripts are placed under ```scripts/mipha/eval/```.
For example, to test the model's performance on VQAv2 dataset, simply run:

```
bash scripts/mipha/eval/vqav2.sh
```

## :crystal_ball: Suppored Capacities (Covering 20 tasks)

![image](https://github.com/yuanze-lin/Olympus/blob/main/asset/capacities.png)


## :snowboarder: Diverse Applications

![image](https://github.com/yuanze-lin/Olympus/blob/main/asset/application.png)

## Citation

If you find Olympus useful for your research and applications, please cite using this BibTeX:

```
@article{lin2024olympus,
  title={Olympus: A Universal Task Router for Computer Vision Tasks},
  author={Lin, Yuanze and Li, Yunsheng and Chen, Dongdong and Xu, Weijian and Clark, Ronald and Torr, Philip HS},
  journal={arXiv preprint arXiv:2412.09612},
  year={2024}
}
```

## Acknowledgement
Our project is built upon the following foundations:

- [Mipha](https://github.com/xmoanvaf/llava-phi): An impressive open-source project for lightweight vision-language assistants
- [LLaVA](https://github.com/haotian-liu/LLaVA): A powerful open-source vision-language assistant project

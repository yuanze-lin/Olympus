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

### Inference

Run the following code for inference: 
```
model_name=Olympus
MODELDIR=ckpts/$model_name

python predict.py \
  --prompt "Generate an image of a fluffy orange cat lounging on a windowsill, \
with sunlight streaming through the glass and casting soft shadows to create a cozy atmosphere. \
Next, would it be possible to change the cat's color to white? This change will make it more eye-catching. \
In the following step, produce a high-resolution 3D model based on the modified image. \
At the next point, please show a video of a cat and a dog running on a playground." \
  --model-path $MODELDIR \
  --temperature 0 \
  --conv-mode v0
```
Alternatively, you can run ```bash predict.sh``` as we did. 

The prediction should be like:
```
Input Prompt:  Generate an image of a fluffy orange cat lounging on a windowsill,
with sunlight streaming through the glass and casting soft shadows to create a cozy atmosphere.
Next, would it be possible to change the cat's color to white? This change will make it more eye-catching.
In the following step, produce a high-resolution 3D model based on the modified image.
At the next point, please show a video of a cat and a dog running on a playground.

Output:  <image_gen>a fluffy orange cat lounging on a windowsill, with sunlight streaming
through the glass and casting soft shadows to create a cozy atmosphere.</image_gen>
<image_edit>change the cat's color to white.</image_edit>
<3D_gen_image>produce a high-resolution 3D model based on the modified image.</3D_gen_image>
<video_gen>a cat and a dog running on a playground.</video_gen>
```
Change the ```--prompt``` to customize the input prompt as needed.

### :rocket: From Routing Tokens to Real Outputs <a href="#specialists" id="specialists"/>

`predict.py` shows you *which* specialist Olympus picked; `run_tools.py` actually
**calls** it, so a single instruction turns into real images, videos and 3D meshes
([#1](https://github.com/yuanze-lin/Olympus/issues/1)).

```
                    ┌──────────────┐   routing tokens   ┌───────────────────┐
 user instruction → │   Olympus    │ ─────────────────▶ │  olympus_tools    │ → PNG / MP4 / GLB
                    │  (router)    │  <image_gen>...    │   30 tokens       │
                    └──────────────┘                    └───────────────────┘
```

#### Install

The router and **every** specialist run in a **single `olympus` environment**:

```
conda activate olympus
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements_tools.txt
bash scripts/install_specialists.sh
```

Weights stream from the Hugging Face Hub on first use. The SOTA defaults are
large (~200 GB if you exercise every token), so point the cache at a big disk:

```
export HF_HOME=/path/to/big/disk/hf_cache
python scripts/prefetch_specialists.py     # optional: pre-download everything
```

#### Quick start

Run the same example as `predict.sh`, but produce the actual assets:

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
automatically** — the edit runs on the generated image, and the 3D model is built
from the edited image:

```
Execution plan:
  [0] <image_gen> via qwen_image
        "a fluffy orange cat lounging on a windowsill, ..."
  [1] <image_edit> via qwen_image_edit  <- step 0
        "change the cat's color to white."
  [2] <3D_gen_image> via image_to_3d  <- step 1
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
├── step2_3D_gen_image.ply / .glb   # open in any 3D viewer
└── step3_video_gen.mp4
```

Measured on a single 48 GB GPU (the four steps above, full quality):

| Step | Model | Time |
|---|---|---|
| `<image_gen>` | Qwen-Image | 131.5 s |
| `<image_edit>` | Qwen-Image-Edit-2509 | 187.8 s |
| `<3D_gen_image>` | Shap-E | 4.7 s |
| `<video_gen>` | Wan2.2-TI2V-5B | 191.5 s |

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
| `<3D_gen_text>` | Text-to-3D | Shap-E | LGM *(custom rasteriser)* |
| `<3D_gen_image>` | Image-to-3D | TripoSR if installed, else Shap-E | Wonder3D *(custom build)* |
| `<{pose,canny,depth,normal,seg,scrib}_to_image>` | Controllable Image Gen | ControlNet (per condition) | ControlNet |
| `<{pose,canny,depth,normal,seg,scrib}_to_video>` | Controllable Video Gen | Text2Video-Zero + ControlNet | Text2Video-Zero |

Which model produced each artifact is recorded per step in `manifest.json`.

> **Licensing note.** Depth Anything 3 checkpoints are released under
> CC-BY-NC-4.0 (non-commercial). For a permissive alternative use
> `--backend-model image_depth=depth-anything/Depth-Anything-V2-Small-hf`
> (Apache-2.0).

#### Notes

- **One GPU is enough.** Specialists load lazily and the previous one is evicted
  before the next is built, so peak VRAM is roughly a single model rather than the
  sum. The router is freed before the first specialist loads. The large models
  (Qwen-Image, Wan2.2) use CPU offload; on a 48 GB card `<image_gen>` takes about
  6 minutes at full quality — lower `--step-option image_gen.steps=15` to trade
  quality for speed.
- **Condition maps are handled for you.** `<canny_to_image>` on a raw photo derives
  the edge map first; chained after `<image_canny>` the upstream map is reused
  instead of being recomputed.
- **Failures are isolated.** A specialist that errors is recorded in
  `manifest.json` and the remaining steps still run.
- **Extending it** takes one class: subclass `Backend`, decorate with
  `@register("my_backend")`, and point a token at it in `olympus_tools/tokens.py`.

#### Dependency notes (please read before upgrading)

Running a 2024 router alongside 2025/2026 specialists in one environment needs a
specific version window. Two fixes make it possible:

1. **`mipha_phi.py` now supplies `position_ids` and handles `Cache` objects.**
   transformers ≥ 4.50 stopped inferring absolute positions from the cache
   length, and hands `generate()` an already-instantiated empty cache, so the old
   `if past_key_values:` test never fired. The result was a *silent* failure: the
   router still produced fluent text, but every token after the first was decoded
   at position 0, so it emitted no routing tokens at all. Olympus now runs
   correctly on modern transformers.
2. **`timm` must be ≥ 1.0.** transformers ≥ 4.49 ships a `timm_wrapper` model that
   imports `timm.data.ImageNetInfo`, added in timm 1.0. `controlnet_aux 0.0.9`
   declares `timm<=0.6.7`, but that pin is stale — the detectors Olympus uses
   (OpenPose, DWPose, NormalBAE, PidiNet) are verified working on timm 1.0.15, so
   install timm afterwards and ignore pip's warning.

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

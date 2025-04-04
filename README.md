<p align="center"><img src="https://github.com/yuanze-lin/Olympus/blob/main/asset/olympus.png" alt="icon" width="150" height="150" style="vertical-align:middle; margin-right:5px;" /></p>

# Olympus: A Universal Task Router for Computer Vision Tasks (CVPR 2025) <br /> 

[![PDF](https://img.shields.io/badge/PDF-Download-orange?style=flat-square&logo=adobeacrobatreader&logoColor=white)](https://arxiv.org/pdf/2412.09612)
[![arXiv](https://img.shields.io/badge/arXiv-2412.09612-b31b1b.svg)](https://arxiv.org/abs/2412.09612)
[![Project Page](https://img.shields.io/badge/Project%20Page-Visit%20Now-0078D4?style=flat-square&logo=googlechrome&logoColor=white)](https://yuanze-lin.me/Olympus_page/)
[![Weights](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E)](https://huggingface.co/Yuanze/Olympus)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-FFD21E)](https://huggingface.co/datasets/Yuanze/Olympus)

Official implementation of "Olympus: A Universal Task Router for Computer Vision Tasks" 

**:hearts: If you find our project is helpful for your research, please kindly give us a :star2: and cite our paper :bookmark_tabs:   : )**

## :mega:  News
- [ ] Release the code for integration with task-specific models.
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

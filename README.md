<p align="center"><img src="https://github.com/yuanze-lin/Olympus/blob/main/asset/olympus.png" alt="icon" width="150" height="150" style="vertical-align:middle; margin-right:5px;" /></p>

# Olympus: A Universal Task Router for Computer Vision Tasks (CVPR 2025) <br /> 

[![PDF](https://img.shields.io/badge/PDF-Download-orange?style=flat-square&logo=adobeacrobatreader&logoColor=white)](https://arxiv.org/pdf/2412.09612)
[![arXiv](https://img.shields.io/badge/arXiv-2412.09612-b31b1b.svg)](https://arxiv.org/pdf/2412.09612) 
[![Project Page](https://img.shields.io/badge/Project%20Page-Visit%20Now-0078D4?style=flat-square&logo=googlechrome&logoColor=white)](https://yuanze-lin.me/Olympus_page/)
[![Weights](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E)](https://huggingface.co/Yuanze/Olympus)

Official implementation of "Olympus: A Universal Task Router for Computer Vision Tasks" 

**:hearts: If you find our project is helpful for your research, please kindly give us a :star2: and cite our paper :bookmark_tabs:   : )**

## :mega:  News
- [ ] Release the training code.
- [ ] Release Olympus datasets.
- [ ] Release the inference code of Olympus.
- [x] We've released the weights of Olympus.

## :page_facing_up: Abstract

> We introduce Olympus, a new approach that transforms Multimodal Large Language Models (MLLMs) into a unified framework capable of handling a wide array of computer vision tasks. Utilizing a controller MLLM, Olympus delegates over 20 specialized tasks across images, videos, and 3D objects to dedicated modules. This instruction-based routing enables complex workflows through chained actions without the need for training heavy generative models. Olympus easily integrates with existing MLLMs, expanding their capabilities with comparable performance. Experimental results demonstrate that Olympus achieves an average routing accuracy of 94.75% across 20 tasks and precision of 91.82% in chained action scenarios, showcasing its effectiveness as a universal task router that can solve a diverse range of computer vision tasks.

## :low_brightness: Overview 

![image](https://github.com/yuanze-lin/Olympus/blob/main/asset/overview.png)


## :crystal_ball: Suppored Capacities (Covering 20 tasks)

![image](https://github.com/yuanze-lin/Olympus/blob/main/asset/capacities.png)


## :snowboarder: Diverse Applications

![image](https://github.com/yuanze-lin/Olympus/blob/main/asset/application.png)

## Citation

If you find our work useful in your research or applications, please consider citing our paper using the following BibTeX:

```
@article{lin2024olympus,
  title={Olympus: A Universal Task Router for Computer Vision Tasks},
  author={Lin, Yuanze and Li, Yunsheng and Chen, Dongdong and Xu, Weijian and Clark, Ronald and Torr, Philip HS},
  journal={arXiv preprint arXiv:2412.09612},
  year={2024}
}
```

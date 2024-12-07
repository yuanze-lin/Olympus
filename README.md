# Olympus: A Universal Task Router for Computer Vision Tasks <br /> 
**:hearts: If you find our project is helpful for your research, please kindly give us a :star2: and cite our paper :bookmark_tabs:   : )**

[![PDF](https://img.shields.io/badge/PDF-Download-orange?style=flat-square&logo=adobeacrobatreader&logoColor=white)](https://openaccess.thecvf.com/content/CVPR2024/papers/Lin_Text-Driven_Image_Editing_via_Learnable_Regions_CVPR_2024_paper.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2412.svg)](https://arxiv.org/pdf/2412) 
[![Project Page](https://img.shields.io/badge/Project%20Page-Visit%20Now-0078D4?style=flat-square&logo=googlechrome&logoColor=white)](https://yuanze-lin.me/Olympus_page/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://colab.research.google.com/drive/1mRWzNOlo_RR_zvrnHODgBoPAa59vGUdz#scrollTo=v_4gmzDOzN98)

Official implementation of "Text-Driven Image Editing via Learnable Regions" 

[Yuanze Lin](https://yuanze-lin.me/), [Yunsheng Li](https://scholar.google.com/citations?user=hJrIyCwAAAAJ&hl=en), [Dongdong Chen](https://www.dongdongchen.bid/), [Weijian Xu](https://weijianxu.com/), [Ronald Clark](https://www.ron-clark.com/), [Philip H. S. Torr](https://eng.ox.ac.uk/people/philip-torr/)


> Abstract: We introduce Olympus, a new approach that transforms Multimodal Large Language Models (MLLMs) into a unified framework capable of handling a wide array of computer vision tasks. Utilizing a controller MLLM, Olympus delegates over 20 specialized tasks across images, videos, and 3D objects to dedicated modules. This instruction-based routing enables complex workflows through chained actions without the need for training heavy generative models. Olympus easily integrates with existing MLLMs, expanding their capabilities with comparable performance. Experimental results demonstrate that Olympus achieves an average routing accuracy of 94.75% across 20 tasks and precision of 91.82% in chained action scenarios, showcasing its effectiveness as a universal task router that can solve a diverse range of computer vision tasks.

## Method 

![image](https://github.com/yuanze-lin/Olympus/blob/main/asset/framework.png)


## :loudspeaker:  News
- [To do] Release the training code.
- [To do] Release the Olympus-Instruct and Olympus-Bench.
- [To do] Release the inference code of Olympus (within 1 week).  
- [2024.12.07] The Olympus model has been released on Hugging Face.

## Suppored Capacities (Covering 20 tasks)

![image](https://github.com/yuanze-lin/Olympus/blob/main/asset/capacities.png)


## Diverse Applications

![image](https://github.com/yuanze-lin/Olympus/blob/main/asset/application.png)

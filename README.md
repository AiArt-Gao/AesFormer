# Mixture-of-Modality-Experts for Unified Image Aesthetic Assessment with Multi-Level Adaptation (Aesformer)

## Abstract
Image aesthetic assessment (IAA) is challenging, because human’s judgement of aesthetic is a metaphysical integration of multi-level information, including color, composition, and semantic. Most existing methods try to learn such information merely from images for the Vision-only IAA (VIAA) task. Recently, a number of Multi-modal IAA (MIAA) methods have been proposed to additionally explore text comments for capturing comprehensive information. However, these MIAA methods are not applicable or show limited performance, when there are no text comments available. To combat this challenge, we propose a unified IAA framework, termed AesFormer, by using mixtures of vision-language Transformers. Specially, AesFormer first learns aligned image-text representations through contrastive learning, and uses a vision-language head for MIAA prediction. Afterward, we propose a multi-level adaptation method to adapt the learned MIAA model to the case without text comments, and use another vision head for VIAA prediction. Extensive experiments are conducted on the AVA, Photo.net, and JAS datasets. The results show that AesFormer significantly outperforms previous methods in both MIAA and VIAA tasks, on all datasets. Remarkably, all the three main metrics, including the classification accuracy, PLCC, and SRCC, break through 90\% for the first time, on the AVA dataset. Our codes and models have been released online at: \url{https://github.com/AiArt-HDU/aesformer}.

## Attention Visualization
<img width="729" alt="image" src="https://github.com/AiArt-HDU/aesformer/assets/101108289/a5fb6839-6ee8-4a5b-a3be-7db35327056e">

<img width="732" alt="image" src="https://github.com/AiArt-HDU/aesformer/assets/101108289/5923b4d8-809e-4c53-80de-100b4f393e73">



## Prerequisites
- Linux or macOS
- Python 3.8
- Pytorch 1.8
- CPU or NVIDIA GPU + CUDA CuDNN

## Pretrained Models
- Aesformer-T : [[Baidu CLoud](https://pan.baidu.com/s/1U1EQyr76-q8AkCvawWz9mQ )] pwd:6guc

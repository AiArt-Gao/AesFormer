# Multi-modal Transformers for Image Aesthetic Assessment with Mixture-of-Modality-Experts and Multi-Scale Adaptation (Aesformer)

## Abstract
Image aesthetic assessment (IAA) is challenging, because human’s judgement of aesthetic is a metaphysical integration of multi-level information, including color, composition, and semantic. Traditional methods merely learn aesthetic representations from images to capture such information. Recently, a number of methods have been proposed to explore comprehensive information from text comments. However, these methods either show limited performance or are not applicable, when there’s no text comments available. To alleviate these deficiencies, in this paper, we propose a novel IAA method by using mixtures of vision-language Transformers. First, we learn multi-modal aesthetic representations through cross-modal contrastive learning, with a vision-language head for aesthetic prediction. Afterwards, we propose a multi-scale adaptation method to adapt the learned model to the case without text comments. Specially, we apply lightweight adapters after multi-scale outputs of the vision stream, and use another vision head for aesthetic prediction. Our full model intelligently switches the prediction branch based on the input, and works well no matter whether text comments are available or not. Experimental results show that our multi-modal model achieves state-of-the-art performance on the AVA dataset, with all the three main metrics, including the classification accuracy, PLCC, and SRCC, breaking through 90% for the first time. Besides, with minimal finetuning cost using the proposed multi-scale adaptation method, our model significantly outperforms previous methods on all benchmark datasets, including the AVA, Photo.net, and JAS datasets. 

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

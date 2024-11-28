
# Image Generation using StyleGAN Pre-trained Model


This project applies the use of StyleGAN in generating high-quality images by using pre-trained models and datasets from Kaggle. The resulting model synthesizes images based on the FFHQ (Flickr-Faces-HQ) dataset.




## Table of contents

Setup Instructions
Model Description
Training Process
Output Visualization
Usage
Acknowledgements
## Setup Instructions


Install my-project with npm

```bash
import kagglehub
greatgamedota_ffhq_face_data_set_path = kagglehub.dataset_download('greatgamedota/ffhq-face-data-set')
songseungwon_ffhq_1024x1024_pretrained_path = kagglehub.dataset_download('songseungwon/ffhq-1024x1024-pretrained')

```


Move Datasets:

```bash
import shutil
shutil.move(greatgamedota_ffhq_face_data_set_path, "/content/ffhq-face-data-set")
shutil.move(songseungwon_ffhq_1024x1024_pretrained_path, "/content/ffhq-pretrained")


```
    
    
Import Required Libraries:

```bash
import torch
import torchvision
import matplotlib.pyplot as plt

```
    
## Model Description

The model uses StyleGAN, which leverages convolutional neural networks for image synthesis. It includes custom layers for enhanced learning
## Training Process


The model synthesizes images from latent vectors using a series of layers:

    1. Linear layers
    2. Convolution layers
    3. Noise layers
    4. StyleMod layers

Pre-trained weights are loaded for efficient training:


```bash
g_all.load_state_dict(torch.load('/content/ffhq-pretrained/karras2019stylegan-ffhq-1024x1024.for_g_all.pt'))

```

## Output Visualization

Generated images are visualized using matplotlib:



```bash
plt.imshow(imgs.permute(1,2,0).detach().numpy())
plt.axis('off')
plt.show()

```


## Usage

    1. Run the notebook step-by-step to download datasets, load pre-trained models, and visualize the outputs.
    2. Customize the latent vector to generate diverse image outputs.

## Acknowledgements

    1. Kaggle: For datasets and pre-trained models.

    2. StyleGAN: Developed by NVIDIA for high-resolution image synthesis.

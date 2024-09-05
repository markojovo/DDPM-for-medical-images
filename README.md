# DDPM for Medical Image Generation

This repository contains my implementation of a Denoising Diffusion Probabilistic Model (DDPM) for generating medical images, specifically chest X-rays.

The model is designed to predict an image $x_t$ given a very slightlier noisy version of the image $x_{t+1}$

with the noise schedule $x_{t+1} := x_t (1 - \alpha_t) + \mathcal{N}(0, 1) \alpha_t$

where $\alpha_t = \alpha(t)$, which follows a specific cosine gain noise schedule from 0 to 1

i. e. a standard denoising diffusion neural network image generator.


## Overview

- **Model**: DDPM with symmetric encoder-decoder architecture
- **Framework**: PyTorch
- **Input/Output**: 128x128 grayscale images

## Key Features

- Cosine noise schedule
- Feature-wise Linear Modulation (FiLM) for time step conditioning
- Squeeze-and-Excitation (SE) attention in convolutional blocks

## Installation

```bash
git clone https://github.com/yourusernameDDPM-for-medical-images.git
```

## Results

The model generates chest X-ray images. While the results show emerging patterns, there's room for improvement in image quality.

## Future Work

- Develop a more specified evaluation pipeline
- Explore application to 1D or multi-channel 1D medical signals

## Acknowledgements

The code here is my contribution of the final class project as part of the CMPT 340 course, taught by Dr. Ghassan Hamarneh.

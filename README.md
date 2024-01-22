# Land-Segmentation
[![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/t17q5l89hk/notebook/r2qjh2d2v9c9ft6?file=%2Fland_segmentation.ipynb)

The project is meant to host the source code for the dissertation project on satellite remote sensing optimisation with different self-supervised pretraining methods.

The current pretraining methods are:
- [x] Masked-Autoencoder
- [x] DINOv2

Current backbones in use:
- [x] ViT-S8 (DINOv2, MAE)
- [ ] HSI optimized backbone

Current segmentation decoder networks in use:
- [x] MS linear head (DINOv2)
- [ ] TBD (MAE - latent space optimized decoders)
- [ ] TBD (DINOv2 - multi-scale optimized decoders)

## Goal of the project
The goal of the project is to implement a state of the art end-to-end training pipeline for hyper-spectral image data processing using existing state of the art methods.
The parts of the project that will be evaluated:
- Pretraining methods, and which one is more suitable(and why) for HSI extraction.
- Decoder networks, and what networks are more suitable for segmentation for the previously used training methods.

In either case the encoders are frozen on the downstream training, for better comparison of the methods.


## Setup
The project runs on the nvidia cuda platform. the easiest way to run the project is through the docker file provided in the repository, which contains all the necessary dependencies for the project.

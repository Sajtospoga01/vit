# Land-Segmentation
[![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/t17q5l89hk/notebook/r2qjh2d2v9c9ft6?file=%2Fland_segmentation.ipynb)

>The project is under huge changes as of now. From the previous CNN based design to ViT based design, and from TF2 to PyTorch. The project is not yet ready for use.

This repository contains the code for my paper on hyperspectral land segmentation in remote sensing.
In this paper we propose a way of pretraining hyperspectral vision transformers.

## Goal of the project
The goal of the project is to implement a state of the art end-to-end training pipeline with the following features:

- [x] ViT based model
- [x] MAE based self-supervised pretraining
- [ ] Finetuning on downstream task for land segmentation

## Setup
The project is inteded to run on the Paperspace Gradient platform however if needed it can be ran locally. The project uses the orbit-software-base as its image, using this image will ensure that all dependencies are installed. The image is gpu enabled such that the model can be trained on a gpu. The image can be found [here](https://hub.docker.com/repository/docker/guorbit/orbit-software-base/general). This includes a propreitary library the orbit utilities for deep learning

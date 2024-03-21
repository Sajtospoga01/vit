# Land-Segmentation

The project is meant to host the source code for the dissertation project on satellite remote sensing optimisation with different self-supervised pretraining methods.

The current pretraining methods are:
- [x] Masked-Autoencoder
- [x] DINOv2

Current backbones in use:
- [x] ViT-S8 (DINOv2, MAE)
- [x] HSI optimized backbone (Spatial spectral backbone)

Current segmentation decoder networks in use:
- [x] Multiscale linear decoder
- [x] Multiscale Convolutional decoder
- [x] Segmenter transformer decoder

## Goal of the project
The goal of the project is to implement a state of the art end-to-end training pipeline for hyper-spectral image data processing using existing state of the art methods.
The parts of the project that will be evaluated:
- Pretraining methods, and which one is more suitable(and why) for HSI extraction.
- Decoder networks, and what networks are more suitable for segmentation for the previously used training methods.

In either case the encoders are frozen on the downstream training, for better comparison of the methods.

## project structure

The project is structured in the following way:
- configs: contains the configuration files for the training of the models
- build: contains the yml files for training jobs
- src: contains the source code for the project
- wiki: contains the documents for the project
- root: contains the docker file, main notebooks, and main training files for the project
- process: contains files that were used in the processing and extra analysis of the data


## Setup
The project runs on the nvidia cuda platform. the easiest way to build and run the project is through the docker file provided in the repository, which contains all the necessary dependencies for the project.
> Note: in order for the project build properly the nvidia gpu is required to compile the some of the dependencies.
> Note: This container is meant to be built in a compute cluster environment as part of a CD setup, such it might require changing the paths to build correctly.

The container exposes the jupyter notebook on port 8888, and the tensorboard on port 6006, so it should be possible to connect to the container from the host machine, however this was never tested.

In order to load weights in the weights directory has to be downloaded from the dissertation onedrive folder, which requires a university account to access (further readme provided in the folder extra information on the files):
https://gla-my.sharepoint.com/:f:/g/personal/2575706b_student_gla_ac_uk/EhmZoHxvQrdCrisbgYtADBEB15e1SN6Q6-N_U02FFJvoxg?e=jqLEgK

In order to access the datasets, it can be done in a similar way (further readme provided in the folder extra information on the files):
https://gla-my.sharepoint.com/:f:/g/personal/2575706b_student_gla_ac_uk/Ei5TaU99a9VNjE5H-HcFNAcBbDqBAGVK6sPxOVISq7eobw?e=bMS6fm

## Running the project

The project has a lot of notebooks associated with it. These were used for prototyping. Actual training was done using the configuration files located in the configs folder, and any code that is marked as train_* or tune_* and is meant to be given as the entry point to the container.

## Documents structure

As the projects documents were originally source controlled in the wiki of the repository, the documents were additionally added to the repository under wiki folder.
Structure:
- wiki
  - timelogs/2575706.rmd
            - week-1,2,...
  - reports/status-report-dec15.md
  - meetings/combined-mettings.md
            - meeting-1,2,...
  - dissertation/4th_Year_dissertation_final.pdf
  - presentation/presentation.pdf
  - plan.md



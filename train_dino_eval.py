import random
import torch
from torch import nn
import numpy as np
import os
import wandb
from src.utils.utils import load_cfg
from src.utils.data_loader import FlowGeneratorExperimental
from src.utils.data_loader_strategy import DataFactoryStrategy
from utilities.segmentation_utils.reading_strategies import BatchReaderStrategy
from utilities.segmentation_utils.constants import (
    FileType,
    ImageOrdering,
)
from utilities.segmentation_utils.ImagePreprocessor import PreprocessingQueue
from src.dinov2.trainer import DINOv2Trainer
from src.dinov2.models.dinov2_model import SSLMetaArch
import src.dinov2.distributed as distributed 
# from src.utils.callbacks import ModelCheckpoint
from mmseg.datasets.builder import build_dataset, build_dataloader
from src.dinov2.eval.dataloader import WHU_OHS, RepositionData,MyLoadImageFromFile
TRAINING_DATA_PATH = "/nfs/datasets/full_data/full/"
VALIDATION_DATA_PATH = "/nfs/datasets/full_data/full/"

NUM_CLASSES = 25

mean_per_band = np.array([
    136.43702139, 136.95781982, 136.70735693, 136.91850906, 137.12465157,
    137.26050865, 137.37743316, 137.24835798, 137.04779119, 136.9453704,
    136.79646442, 136.68328908, 136.28231996, 136.02395119, 136.01146934,
    136.72767901, 137.38975674, 137.58604882, 137.61197314, 137.46675538,
    137.57319831, 137.69239868, 137.72318172, 137.76894864, 137.74861655,
    137.77535075, 137.80038781, 137.85482571, 137.88595859, 137.9490434,
    138.00128494, 138.17846624
])
std_per_band = np.array([
    33.48886853, 33.22482796, 33.4670978, 33.53758141, 33.48675988, 33.33348355,
    33.35096189, 33.63958817, 33.85081288, 34.08314358, 34.37542553, 34.60344274,
    34.80732573, 35.17761688, 35.1956623, 34.43121367, 33.76600779, 33.77061146,
    33.92844916, 34.0370747, 34.0285642, 33.87601205, 33.81035869, 33.66611756,
    33.74440912, 33.69755911, 33.69845938, 33.6707364, 33.62571536, 33.44615438,
    33.27907802, 32.90732107
])
HPARAMS = {
    'mini_batch_size': 128,
    'batch_size': 256,
    'epoch': 100,
    'criterion': nn.MSELoss,
    'optimizer': torch.optim.Adam,
    'optimizer_params': {
        'lr': 0.0001,
    },

    # loader parameters
    'shuffle': True,
    'preprocess': True,

    # model paramters
    "model_parameters": {
        "patch_size": 8,
        # encoder specific params
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        # decoder specific params
        "decoder_embed_dim": 512,
        "decoder_depth": 8,
        "decoder_num_heads": 16,
        "mlp_ratio": 4,
    },

    "io_params": {
        "input_size": (512, 512),
        "bands": 32,
        "output_size": (512, 512),
        "num_classes": NUM_CLASSES,
    },
}

   
def load_weigths_to_model(model,weights):
    # load the weights from the checkpoint into the model
    new_state_dict = model.state_dict()
    print(new_state_dict.keys())
    old_state_dict = torch.load(weights)
    print(old_state_dict.keys())
    for name, param in old_state_dict.items():
        if name in new_state_dict and param.size() == new_state_dict[name].size():
            print("Loading layer: ", name)
            new_state_dict[name].copy_(param)
            param.requires_grad = False
        else:
            # Handle layers that do not match or additional processing if required
            print(f"Skipping layer: {name}, as it's not present or mismatched in the new model")
            print(param.size(), new_state_dict[name].size())
            
    
    # Load the modified state dict into the new model
    model.load_state_dict(new_state_dict, strict=False)
    return model    
    



def main():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    wandb.login()

    distributed.enable(overwrite=True)

    cfg = load_cfg()

    datasets = [
        build_dataset(
            dict(
                type='WHU_OHS',
                data_root=TRAINING_DATA_PATH,
                img_dir='images/train',
                ann_dir='images/validate/',
                        pipeline=[
            dict(type='MyLoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=True),

            dict(type='RepositionData'),

            dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]
            )
        ),
    ]


    data_loaders = [
        build_dataloader(
            ds,
            samples_per_gpu = 8,
            workers_per_gpu = 1,
            seed = 42,
            drop_last = True,
            
        ) for ds in datasets
    ]



    Dino2ModelHandler = SSLMetaArch(cfg)
    load_weigths_to_model(Dino2ModelHandler,"/nfs/dinov2_checkpoint.pth")
    total_params = sum(p.numel() for p in Dino2ModelHandler.parameters())
    print(f'{total_params:,} total parameters.')
    print(f"Separate sizes:\n\tTeacher: {sum(p.numel() for p in Dino2ModelHandler.teacher.parameters()):,}\n\tStudent: {sum(p.numel() for p in Dino2ModelHandler.student.parameters()):,}")

    optim = torch.optim.AdamW(Dino2ModelHandler.get_params_groups(), betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))
    dino_trainer = DINOv2Trainer(
        cfg=cfg,
        model=Dino2ModelHandler,
        optimizer=optim,
        criterion=None,
        train_loader=data_loaders[0],
        device=torch.device("cuda:0"),
    )
    callbacks = [
        # ModelCheckpoint(
        #     "/nfs/dinov2_checkpoint.pth",
        #     monitor="loss",
        #     mode="min",
        # )
    ]
    dino_trainer.train(callbacks=callbacks)

if __name__ == "__main__":
    main()
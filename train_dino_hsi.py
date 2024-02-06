import random
import torch
from torch import nn
import numpy as np
import os
import wandb
from keys import load_env
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
from src.dinov2.models.dinov2_model import SSLMetaArchHSI as SSLMetaArch
import src.dinov2.distributed as distributed 
# from src.utils.callbacks import ModelCheckpoint


TRAINING_DATA_PATH = "/nfs/datasets/bchsi/pb_tr/"
VALIDATION_DATA_PATH = "/nfs/datasets/bchsi/pb_val/"

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
        "input_size": (64, 64),
        "bands": 32,
        "output_size": (64, 64),
        "num_classes": NUM_CLASSES,
    },
}




def main():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    load_env()
    wandb.login()

    distributed.enable(overwrite=True)

    cfg = load_cfg()

    in_queue = PreprocessingQueue([])

    val_queue = PreprocessingQueue([])

    X_train = BatchReaderStrategy(
        os.path.join(TRAINING_DATA_PATH, "image"),
        image_size=HPARAMS['io_params']['input_size'],
    )

    X_val = BatchReaderStrategy(
        os.path.join(VALIDATION_DATA_PATH, "image"),
        image_size=HPARAMS['io_params']['input_size'],
    )

    batch_reader = DataFactoryStrategy(X_train)

    batch_reader_val = DataFactoryStrategy(X_val)

    reader_args = {
        "input_strategy": batch_reader,
        "output_strategy": batch_reader,
        "shuffle": HPARAMS["shuffle"],
        "preprocessing_enabled": HPARAMS["preprocess"],
        "channel_mask": [True for _ in range(32)],
        "num_classes": NUM_CLASSES,
        "batch_size": HPARAMS["batch_size"],
        "image_ordering": ImageOrdering.CHANNEL_FIRST,
        "type": [FileType.MULTICHANNEL, FileType.MULTICHANNEL],
        "preprocessing_queue_image": in_queue,
        "preprocessing_queue_mask": in_queue,
    }

    val_reader_args = {
        "input_strategy": batch_reader_val,
        "output_strategy": batch_reader_val,
        "shuffle": False,
        "preprocessing_enabled": True,
        "channel_mask": [True for _ in range(32)],
        "num_classes": NUM_CLASSES,
        "batch_size": HPARAMS["batch_size"],
        "image_ordering": ImageOrdering.CHANNEL_FIRST,
        "type": [FileType.MULTICHANNEL, FileType.MULTICHANNEL],
        "preprocessing_queue_image": val_queue,
        "preprocessing_queue_mask": val_queue,
    }

    reader = FlowGeneratorExperimental(**reader_args)
    val_reader = FlowGeneratorExperimental(**val_reader_args)
    reader.set_mini_batch_size(HPARAMS["mini_batch_size"])
    val_reader.set_mini_batch_size(HPARAMS["mini_batch_size"])

    Dino2ModelHandler = SSLMetaArch(cfg)

    total_params = sum(p.numel() for p in Dino2ModelHandler.parameters())
    print(f'{total_params:,} total parameters.')
    print(f"Separate sizes:\n\tTeacher: {sum(p.numel() for p in Dino2ModelHandler.teacher.parameters()):,}\n\tStudent: {sum(p.numel() for p in Dino2ModelHandler.student.parameters()):,}")

    optim = torch.optim.AdamW(Dino2ModelHandler.get_params_groups(), betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))
    dino_trainer = DINOv2Trainer(
        cfg=cfg,
        model=Dino2ModelHandler,
        optimizer=optim,
        criterion=None,
        train_loader=reader,
        device=torch.device("cuda:0"),
        validation_loader=val_reader,
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
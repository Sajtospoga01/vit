import mmcv
import torch
import src.dinov2.distributed as distributed 
import src.dinov2.eval.dataloader
import src.dinov2.eval.optimizers 
import src.dinov2.eval.segmentation_m2f.models.segmentors
from src.dinov2.models.dinov2_model import SSLMetaArch
from src.utils.utils import load_cfg
import wandb

from mmcv.runner import load_checkpoint,build_runner
from mmcv.runner import TextLoggerHook
from mmseg.apis import init_segmentor, inference_segmentor
from mmseg.models import build_segmentor
from mmcv.utils.logging import get_logger
from mmseg.datasets.builder import build_dataset, build_dataloader
import urllib
from mmcv.parallel import collate
from pprint import pprint
from mmseg.core.builder import OPTIMIZER_BUILDERS
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmseg.core.optimizers import layer_decay_optimizer_constructor
from mmseg.core.builder import build_optimizer
from src.dinov2.eval.segmentation import models
from functools import partial
import src.dinov2.eval.segmentation.hooks
from src.vit_model.mae import MaskedAutoencoderViT
import torch.nn as nn
import itertools
import torch.nn.functional as F
import math
NUM_CLASSES = 24
HPARAMS = {
    'mini_batch_size': 32,
    'batch_size': 256,
    'epoch': 30,
    'criterion':nn.MSELoss,
    'optimizer': torch.optim.Adam,
    'optimizer_params': {
        'lr': 0.0001,
    },

    # loader parameters
    'shuffle': True,
    'preprocess': True,

    # model paramters
    "model_parameters":{
        "patch_size": 8,
        #encoder specific params
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        #decoder specific params
        "decoder_embed_dim":512,
        "decoder_depth": 8,
        "decoder_num_heads": 16,
        "mlp_ratio": 4,
    },
    
    "io_params": {
        "input_size": (64,64),
        "bands": 32,
        "output_size": (64,64),
        "num_classes": NUM_CLASSES,
    },
}
class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output

DATASET_PATH = "/nfs/datasets/new_dataset/"
def load_weigths_to_model(model):
    # load the weights from the checkpoint into the model
    new_state_dict = model.state_dict()
    print(new_state_dict.keys())
    old_state_dict = torch.load("/nfs/vit_pretrain/Land-Segmentation/best_hsi_mae_base_spatial.pth")
    print(old_state_dict.keys())
    for name, param in old_state_dict.items():
        if name in new_state_dict and param.size() == new_state_dict[name].size():
            print("Loading layer: ", name)
            new_state_dict[name].copy_(param)
            new_state_dict[name].requires_grad = False
        else:
            # Handle layers that do not match or additional processing if required
            print(f"Skipping layer: {name}, as it's not present or mismatched in the new model")
            print(param.size(), new_state_dict[name].size())
            
    
    # Load the modified state dict into the new model
    model.load_state_dict(new_state_dict, strict=False)
    return model    
    

def main():
    wandb.login()
    distributed.enable(overwrite=True)
 
    # load pretrained backbone


    autoencoder = MaskedAutoencoderViT(
        img_size=HPARAMS['io_params']['input_size'],
        in_chans=HPARAMS['io_params']['bands'],
        **HPARAMS["model_parameters"],
    )


    total_params = sum(p.numel() for p in autoencoder.parameters())
    print(f'{total_params:,} total parameters.')
 
    autoencoder = load_weigths_to_model(autoencoder)

    def load_config_from_url(url: str) -> str:
        with urllib.request.urlopen(url) as f:
            return f.read().decode()
    
    def load_config_from_file(path: str) -> str:
        with open(path) as f:
            return f.read()

    cfg_str = load_config_from_file("configs/normal_segmentor_cluster_cfg.py")
    cfg_mmcv = mmcv.Config.fromstring(cfg_str, file_format=".py")
    logger = get_logger("mmcv")
    model = build_segmentor(cfg_mmcv.model)
    model.backbone.forward = partial(
        autoencoder.get_intermediate_layers,
        n = cfg_mmcv.model.backbone.out_indices,
        reshape = True,
    )
    if hasattr(model.backbone, "patch_size"):
        model.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(model.backbone.patch_size)(x[0]))
    model.init_weights()

    # Copy the weights from the pretrained model
    
    # prepare training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    autoencoder.to(device)
    print("Device: ",device)
    model = MMDataParallel(model, device_ids=[0])  # Single GPU

    # Move your model to the selected device

    datasets = [
        build_dataset(cfg_mmcv.data.train)
    ]

    data_loaders = [
        build_dataloader(
            ds,
            samples_per_gpu = 32,
            workers_per_gpu = 1,
            seed = 42,
            drop_last = True,
            
        ) for ds in datasets
    ]
    cfg_mmcv.log_config.hooks[1].init_kwargs.config = cfg_mmcv

    optimizer = build_optimizer(model, cfg_mmcv.optimizer)

    # criterion 

    runner = build_runner(cfg_mmcv.runner, default_args=dict(
        type="IterBasedRunner",
        model=model,
        logger=logger,
        optimizer = optimizer,
        work_dir=cfg_mmcv.work_dir,
        # meta=cfg.get('meta', None)
        )
    )

    runner.register_training_hooks(
        lr_config=cfg_mmcv.lr_config,
        optimizer_config=cfg_mmcv.optimizer_config,
        checkpoint_config=cfg_mmcv.checkpoint_config,
        log_config=cfg_mmcv.log_config)
  
    runner.run(
        data_loaders=data_loaders,
        workflow=cfg_mmcv.workflow,
    )






if __name__ == '__main__':
    main()
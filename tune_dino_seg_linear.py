import mmcv
import torch
import src.dinov2.distributed as distributed 
import src.dinov2.eval.dataloader
import src.dinov2.eval.optimizers 
import src.dinov2.eval.segmentation_m2f.models.segmentors
from src.dinov2.models.dinov2_model import SSLMetaArch, SSLMetaArchHSI
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
from src.dinov2.models import decoder
from mmseg.models.losses import focal_loss
from mmcv.runner import EvalHook, DistEvalHook
from src.dinov2.eval.patchwise_loss import PatchWiseCrossEntropyLoss
from src.dinov2.fsdp import FSDPCheckpointer
import numpy as np
import random

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

DATASET_PATH = "/nfs/datasets/ne_dataset/"
def load_weigths_to_model(model):
    # load the weights from the checkpoint into the model
    new_state_dict = model.state_dict()
    print(new_state_dict.keys())
    old_state_dict = torch.load("/nfs/vit_pretrain/Land-Segmentation/dinov2_checkpoint.pth")
    print(old_state_dict.keys())
    for name, param in old_state_dict.items():
        if name in new_state_dict and param.size() == new_state_dict[name].size():
            print("Loading layer: ", name)
            new_state_dict[name].copy_(param)
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
    # load pretrained backbone
    Dino2ModelHandler = SSLMetaArchHSI(cfg)
    print(cfg.student.num_register_tokens)
    total_params = sum(p.numel() for p in Dino2ModelHandler.parameters())
    print(f'{total_params:,} total parameters.')
    print(f"Separate sizes:\n\tTeacher: {sum(p.numel() for p in Dino2ModelHandler.teacher.parameters()):,}\n\tStudent: {sum(p.numel() for p in Dino2ModelHandler.student.parameters()):,}")
    

    def load_config_from_url(url: str) -> str:
        with urllib.request.urlopen(url) as f:
            return f.read().decode()
    
    def load_config_from_file(path: str) -> str:
        with open(path) as f:
            return f.read()

    cfg_str = load_config_from_file("configs/normal_segmentor_cluster_cfg.py")
    cfg_mmcv = mmcv.Config.fromstring(cfg_str, file_format=".py")
    logger = get_logger("mmcv")
   
    optim = torch.optim.AdamW(Dino2ModelHandler.get_params_groups(), betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2)) # not used only for loading the model back in
    
    fsdp_checkpointer = FSDPCheckpointer(
                Dino2ModelHandler,
                save_dir=cfg.train.output_dir,
                save_to_disk=True,
                optimizer=optim,)

    Dino2ModelHandler = Dino2ModelHandler.to("cuda")

    Dino2ModelHandler.prepare_for_distributed_training()
    fsdp_checkpointer.load("/nfs/model_final.rank_0.pth")
    for param in Dino2ModelHandler.parameters():
        param.requires_grad = False

    # for param in Dino2ModelHandler.parameters():
    #     if param.dtype == torch.float16:
    #         param.data = param.data.to(torch.float32)
    model = build_segmentor(cfg_mmcv.model)
    model.backbone.forward = partial(
        Dino2ModelHandler.teacher.backbone.get_intermediate_layers,
        n = cfg_mmcv.model.backbone.out_indices,
        reshape = True,
    )
    if hasattr(model.backbone, "patch_size"):
        model.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(model.backbone.patch_size)(x[0]))
    model.init_weights()

    # Copy the weights from the pretrained model
    total_params = sum(p.numel() for p in model.backbone.parameters())
    print(f'{total_params:,} total parameters.')

    new_state_dict = model.state_dict()
    old_state_dict = Dino2ModelHandler.teacher.state_dict()

    populated_layers = []

    for name, param in old_state_dict.items():
        param.requires_grad = False

    # prepare training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    Dino2ModelHandler.to(device)
    print("Device: ",device)
    model = MMDataParallel(model, device_ids=[0])  # Single GPU

    # Move your model to the selected device

    datasets = [
        build_dataset(cfg_mmcv.data.train),   
    ]

    eval_dataset = build_dataset(cfg_mmcv.data.val)

    data_loaders = [
        build_dataloader(
            ds,
            samples_per_gpu = 16,
            workers_per_gpu = 1,
            seed = 42,
            drop_last = True,
            
        ) for ds in datasets
    ]

    eval_data_loader = build_dataloader(
        eval_dataset,
        samples_per_gpu = 8,
        workers_per_gpu = 1,
        seed = 42,
        drop_last = True,
    )



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

    eval_hook = EvalHook(eval_data_loader,by_epoch = False, interval = cfg_mmcv.evaluation.interval,save_best = 'mIoU', metric = cfg_mmcv.evaluation.metric, pre_eval = cfg_mmcv.evaluation.pre_eval)

    runner.register_training_hooks(
        lr_config=cfg_mmcv.lr_config,
        optimizer_config=cfg_mmcv.optimizer_config,
        checkpoint_config=cfg_mmcv.checkpoint_config,
        log_config=cfg_mmcv.log_config
            
    )
    
    runner.register_hook(eval_hook)
  
    runner.run(
        data_loaders=data_loaders,
        workflow=cfg_mmcv.workflow,
    )

if __name__ == '__main__':
    main()
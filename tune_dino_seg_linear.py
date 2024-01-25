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
from src.dinov2.models import decoder
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
    wandb.login()
    distributed.enable(overwrite=True)
    cfg = load_cfg()
    # load pretrained backbone
    Dino2ModelHandler = SSLMetaArch(cfg)
    print(cfg.student.num_register_tokens)
    total_params = sum(p.numel() for p in Dino2ModelHandler.parameters())
    print(f'{total_params:,} total parameters.')
    print(f"Separate sizes:\n\tTeacher: {sum(p.numel() for p in Dino2ModelHandler.teacher.parameters()):,}\n\tStudent: {sum(p.numel() for p in Dino2ModelHandler.student.parameters()):,}")
    Dino2ModelHandler = load_weigths_to_model(Dino2ModelHandler)

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

    manual_mapping = {
        "backbone.level_embed":"backbone.blocks.0.ls1.gamma",
        "backbone.blocks.0.gamma1":"backbone.blocks.0.ls1.gamma",
        "backbone.blocks.0.gamma2":"backbone.blocks.0.ls2.gamma",
        "backbone.blocks.1.gamma1":"backbone.blocks.1.ls1.gamma",
        "backbone.blocks.1.gamma2":"backbone.blocks.1.ls2.gamma",
        "backbone.blocks.2.gamma1":"backbone.blocks.2.ls1.gamma",
        "backbone.blocks.2.gamma2":"backbone.blocks.2.ls2.gamma",
        "backbone.blocks.3.gamma1":"backbone.blocks.3.ls1.gamma",
        "backbone.blocks.3.gamma2":"backbone.blocks.3.ls2.gamma",
        "backbone.blocks.4.gamma1":"backbone.blocks.4.ls1.gamma",
        "backbone.blocks.4.gamma2":"backbone.blocks.4.ls2.gamma",
        "backbone.blocks.5.gamma1":"backbone.blocks.5.ls1.gamma",
        "backbone.blocks.5.gamma2":"backbone.blocks.5.ls2.gamma",
        "backbone.blocks.6.gamma1":"backbone.blocks.6.ls1.gamma",
        "backbone.blocks.6.gamma2":"backbone.blocks.6.ls2.gamma",
        "backbone.blocks.7.gamma1":"backbone.blocks.7.ls1.gamma",
        "backbone.blocks.7.gamma2":"backbone.blocks.7.ls2.gamma",
        "backbone.blocks.8.gamma1":"backbone.blocks.8.ls1.gamma",
        "backbone.blocks.8.gamma2":"backbone.blocks.8.ls2.gamma",
        "backbone.blocks.9.gamma1":"backbone.blocks.9.ls1.gamma",
        "backbone.blocks.9.gamma2":"backbone.blocks.9.ls2.gamma",
        "backbone.blocks.10.gamma1":"backbone.blocks.10.ls1.gamma",
        "backbone.blocks.10.gamma2":"backbone.blocks.10.ls2.gamma",
        "backbone.blocks.11.gamma1":"backbone.blocks.11.ls1.gamma",
        "backbone.blocks.11.gamma2":"backbone.blocks.11.ls2.gamma",
    }

    for name, param in old_state_dict.items():
        if name in new_state_dict and param.size() == new_state_dict[name].size():
            # Load the state dict and freeze the layer
            new_state_dict[name].copy_(param)
            param.requires_grad = False
            populated_layers.append(name)
        else:
            # Handle layers that do not match or additional processing if required
            print(f"Skipping layer: {name}, as it's not present or mismatched in the new model")
            print(f"Layer present in new model: {name in new_state_dict}")
            print(param.size(), old_state_dict[name].size())

    # Now we freeze the populated layers by setting requires_grad to False
    for name, param in new_state_dict.items():
        if name in populated_layers:
            param.requires_grad = False
        elif name not in populated_layers and name in manual_mapping:
            # Assuming manual mapping is needed for some layers
            print(f"Loading layer: {name} from {manual_mapping[name]}")
            new_state_dict[name].copy_(old_state_dict[manual_mapping[name]])
            param.requires_grad = False  # Freeze this layer as well
            populated_layers.append(name)
        else:
            print(f"Layer not populated and no manual mapping provided: {name}")


    # prepare training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    Dino2ModelHandler.to(device)
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
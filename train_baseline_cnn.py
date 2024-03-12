import torch
from torch import nn
from mmcv import Config
import wandb
from mmseg.models import build_segmentor
from mmcv.utils.logging import get_logger
from mmseg.datasets.builder import build_dataset, build_dataloader
from mmseg.core.builder import build_optimizer
from mmcv.runner import load_checkpoint,build_runner
from mmcv.parallel import MMDataParallel
from mmcv.runner import EvalHook
import mmcv

def main():
    pass

def load_config_from_file(path: str) -> str:
    with open(path) as f:
        return f.read()

if __name__ == "__main__":
    wandb.login()

    config_file = 'configs/baseline_segmentor_cluster_cfg.py' 
    config = load_config_from_file(config_file)
    config_file = mmcv.Config.fromstring(config,file_format = 'py')
    logger = get_logger("mmcv")
    # Model initialization with checkpoint loading
    model = build_segmentor(config_file, 
                            train_cfg=None, # Assuming you don't want to train immediately
                            test_cfg=config_file.test_cfg, 
                            init_cfg=dict(type='Pretrained', checkpoint='checkpoints/deeplabv3plus/checkpoint.pth')) 

    # Dataset and dataloader initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model = MMDataParallel(model, device_ids=[0])  # Single GPU

    # Move your model to the selected device

    datasets = [
        build_dataset(config_file.data.train),   
    ]

    eval_dataset = build_dataset(config_file.data.val)

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
    

    config_file.log_config.hooks[1].init_kwargs.config = config_file

    optimizer = build_optimizer(model, config_file.optimizer)


    # criterion 

    runner = build_runner(config_file.runner, default_args=dict(
        type="IterBasedRunner",
        model=model,
        logger=logger,
        optimizer = optimizer,
        work_dir=config_file.work_dir,
        # meta=cfg.get('meta', None)
        )
    )

    eval_hook = EvalHook(eval_data_loader,by_epoch = False, interval = config_file.evaluation.interval,save_best = 'mIoU', metric = config_file.evaluation.metric, pre_eval = cfg_mmcv.evaluation.pre_eval)

    runner.register_training_hooks(
        lr_config=config_file.lr_config,
        optimizer_config=config_file.optimizer_config,
        checkpoint_config=config_file.checkpoint_config,
        log_config=config_file.log_config
            
    )
    
    runner.register_hook(eval_hook)
  
    runner.run(
        data_loaders=data_loaders,
        workflow=config_file.workflow,
    )
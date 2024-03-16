
import os
import logging
from src.dinov2.utils import apply_optimizer_scheduler, CosineScheduler,MaskingGenerator
from src.utils.metrics import SSIMAccuracy, Loss, Accuracy
from tqdm import tqdm
from src.dinov2.logging import MetricLogger
from src.utils.utils import CollateDataAndCast
from src.utils.data_loader_strategy import DINOv2DataFactory
from src.dinov2 import distributed
import torch
import math
import wandb
from fvcore.common.checkpoint import PeriodicCheckpointer
from src.dinov2.fsdp import FSDPCheckpointer

logger = logging.getLogger("dinov2")

class DINOv2Trainer():
    def __init__(self,cfg,model,criterion,optimizer,train_loader,device = 'cuda',validation_loader = None) -> None:
        self.model = model.to(device)
        self.model.prepare_for_distributed_training()
        self.criterion = criterion
        self.optimizer = optimizer
        self.validation_loader = validation_loader
        self.cfg = cfg
        self.device = device
        self.loss_metric = Loss()
        self.ssim_accuracy_metric = Accuracy()
        self.val_loss_metric = Loss()
        self.val_ssim_accuracy_metric = SSIMAccuracy()
        self.train_loader = train_loader
        total_steps = cfg.train.epochs * len(self.train_loader)
        metrics_file = os.path.join(self.cfg.train.output_dir, "training_metrics.json")
        metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)

        base_lr = cfg.optim.base_lr
        self.cfg.optim.lr = base_lr
        self.cfg.optim.lr *= math.sqrt(cfg.train.mini_batch_size * distributed.get_global_size() / 1024.0)
        print(f"Global size: {distributed.get_global_size()}")
        print(f"LR: {self.cfg.optim.lr}")
        (
            self.lr_schedule,
            self.wd_schedule,
            self.momentum_schedule,
            self.teacher_temp_schedule,
            self.last_layer_lr_schedule,
        ) = self.__build_schedules(total_steps)
        self.transform = DINOv2DataFactory(
            cfg.crops.global_crops_scale,
            cfg.crops.local_crops_scale,
            cfg.crops.local_crops_number,
            cfg.crops.global_crops_size,
            cfg.crops.local_crops_size,
        )

        inputs_dtype = torch.float32

        img_size = cfg.crops.global_crops_size
        patch_size = cfg.student.patch_size
        n_tokens = (img_size // patch_size) ** 2
        bands = cfg.io.bands
        spectral_patch_size = cfg.student.spectral_patch_size

        n_spectral_tokens = bands//spectral_patch_size
        mask_generator = MaskingGenerator(
            input_size=(img_size // patch_size, img_size // patch_size),
            max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
        )

        mask_spectral_generator = MaskingGenerator(
            input_size=(bands // spectral_patch_size,1),
            max_num_patches=0.5 * bands // spectral_patch_size * 1,
        )

        self.collate_fn = CollateDataAndCast(
            mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
            mask_probability=cfg.ibot.mask_sample_probability,
            n_tokens=n_tokens,
            n_spectral_tokens=n_spectral_tokens,
            mask_generator=mask_generator,
            spectral_mask_generator=mask_spectral_generator,
            dtype=inputs_dtype,
        )
        self.periodic_checkpointer = PeriodicCheckpointer(
            FSDPCheckpointer(
                model,
                save_dir=cfg.train.output_dir,
                save_to_disk=True,
                optimizer=optimizer,
            ),
            period = 1 * len(train_loader),
            max_iter=cfg.train.epochs * len(train_loader),
            max_to_keep=3,
        )
       
       



    def __build_schedules(self,total_steps):
        OFFICIAL_EPOCH_LENGTH = len(self.train_loader)
        lr = dict(
            base_value=self.cfg.optim.lr,
            final_value=self.cfg.optim.final_lr,
            total_iters=self.cfg.train.epochs * OFFICIAL_EPOCH_LENGTH,
            warmup_iters=self.cfg.train.warmup_epochs * OFFICIAL_EPOCH_LENGTH,
            start_warmup_value=0,
        )
        wd = dict(
            base_value=self.cfg.optim.base_wd,
            final_value=self.cfg.optim.final_wd,
            total_iters=self.cfg.train.epochs * OFFICIAL_EPOCH_LENGTH,
        )
        momentum = dict(
            base_value=self.cfg.teacher.base_m_teacher,
            final_value=self.cfg.teacher.final_m_teacher,
            total_iters=self.cfg.train.epochs * OFFICIAL_EPOCH_LENGTH,
        )
        teacher_temp = dict(
            base_value=self.cfg.teacher.teacher_temp,
            final_value=self.cfg.teacher.teacher_temp,
            total_iters=self.cfg.teacher.warmup_teacher_temp_epochs * OFFICIAL_EPOCH_LENGTH,
            warmup_iters=self.cfg.teacher.warmup_teacher_temp_epochs * OFFICIAL_EPOCH_LENGTH,
            start_warmup_value=self.cfg.teacher.warmup_teacher_temp,
        )

        lr_schedule = CosineScheduler(**lr)
        wd_schedule = CosineScheduler(**wd)
        momentum_schedule = CosineScheduler(**momentum)
        teacher_temp_schedule = CosineScheduler(**teacher_temp)
        last_layer_lr_schedule = CosineScheduler(**lr)

        last_layer_lr_schedule.schedule[
            : self.cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
        ] = 0  # mimicking the original schedules

        logger.info("Schedulers ready.")

        return (
            lr_schedule,
            wd_schedule,
            momentum_schedule,
            teacher_temp_schedule,
            last_layer_lr_schedule,
        )

    def __train_step(self,X,y,iter) -> None:
        # Query schedules to be used

        lr = self.lr_schedule[iter]
        wd = self.wd_schedule[iter]
        momentum = self.momentum_schedule[iter]
        teacher_temp = self.teacher_temp_schedule[iter]
        last_layer_lr = self.last_layer_lr_schedule[iter]
        apply_optimizer_scheduler(self.optimizer,lr,wd,last_layer_lr)

        # Forward pass
        self.optimizer.zero_grad(set_to_none=True)
        X = X.data[0]
        X = X.to(self.device)
        X = self.transform(X)
        X = self.collate_fn(X)
        loss_dict = self.model.forward_backward(X, teacher_temp = teacher_temp)

        if self.model.fp16_scaler is not None:
            if self.cfg.optim.clip_grad:
                self.model.fp16_scaler.unscale_(self.optimizer)
                for v in self.model.student.values():
                    v.clip_grad_norm_(self.cfg.optim.clip_grad)
            self.model.fp16_scaler.step(self.optimizer)
            self.model.fp16_scaler.update()
        else:
            if self.cfg.optim.clip_grad:
                for v in self.model.student.values():
                    v.clip_grad_norm_(self.cfg.optim.clip_grad)
            self.optimizer.step()
        self.model.update_teacher(momentum)

        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

        if math.isnan(sum(loss_dict_reduced.values())):
            logger.info("NaN detected")
            raise AssertionError
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        self.loss_metric.update(losses_reduced)
        self.periodic_checkpointer.step(iter)
        # logging


    def __eval_step(self,X,y) -> None:
        pass

    def __push_logs(self,iter) -> None:
        wandb.log(
            {
                "loss": self.loss_metric.compute(),
                "lr": self.lr_schedule[iter],
                "wd": self.wd_schedule[iter],
                "momentum": self.momentum_schedule[iter],
                "last_layer_lr": self.last_layer_lr_schedule[iter],
            }
        )

    def train(self, callbacks = None) -> None:
        epochs = self.cfg.train.epochs
        if "CL_WANDB_PROJECT" in os.environ:
            project = os.environ["CL_WANDB_PROJECT"]
            print(f"Loaded project {project}")
        else:
            project = "vit-dino"

        wandb.init(
            # set the wandb project where this run will be logged
            project=project,
            # track hyperparameters and run metadata
            config=self.cfg,
        )
        for callback in callbacks:
            callback.add_model(self.model)
            callback.on_train_start()


        for epoch in range(epochs):
            print(f"Epoch {epoch}/{epochs}")
            for callback in callbacks:
                callback.on_epoch_start(epoch)

            tqdm_instance = tqdm(
                self.train_loader,
                bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]{desc}",
            )

            for i,data in enumerate(tqdm_instance):
                # break
                for callback in callbacks:
                    callback.on_batch_stairt()

                X = data['img']
                y = data['gt_semantic_seg']

                iter = epoch * len(self.train_loader) + i
                self.__train_step(X,y,iter)
                
                tqdm_instance.set_description(f"loss: {self.loss_metric.compute():.4f}")
               
                self.__push_logs(iter)

                for callback in callbacks:
                    callback.on_batch_end()
            


            logs = {
                "loss": self.loss_metric.compute(),
 
            }
            # if self.validation_loader is not None:
            #     tqdm_instance_val = tqdm(
            #         self.validation_loader,
            #         bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]{desc}",
            #     )
            #     for X,y in tqdm_instance_val:
            #         self.__eval_step(X,y)
        
        
            for callback in callbacks:
                callback.on_epoch_end(epoch,logs)
            
            if epoch == 3:
                print("Unfreezing layers")
                for param in self.model.parameters():
                    param.requires_grad = True
                  
        

        for callback in callbacks:
            callback.on_train_end()

                
            

    def predict(self) -> None:
        pass

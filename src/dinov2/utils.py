import numpy as np
from tqdm import tqdm
import logging
import math
import random

logger = logging.getLogger("dinov2")

def apply_optimizer_scheduler(optimizer, lr, wd, last_layer_lr):
    for param in optimizer.param_groups:
        is_last_layer = param['is_last_layer']
        lr_mulitplier = param['lr_multiplier']
        wd_mulitplier = param['wd_multiplier']
        param['weight_decay'] = wd * wd_mulitplier
        if is_last_layer:
            param['lr'] = last_layer_lr * lr_mulitplier
        else:
            param['lr'] = lr * lr_mulitplier

            
class CosineScheduler:
    def __init__(self, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, freeze_iters=0):
        self.start_lr = base_value
        self.end_lr = final_value
        self.total_steps = total_iters

        freeze_schedule = np.zeros((freeze_iters))
        warmup_schedule = np.linspace(start_warmup_value,base_value,warmup_iters)

        iters = np.arange(total_iters - freeze_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((freeze_schedule,warmup_schedule,schedule))

        assert len(self.schedule) == total_iters

    def __getitem__(self, index):
        if index>= self.total_steps:
            return self.end_lr
        else:
            return self.schedule[index]



class MaskingGenerator:
    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches=0):
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask


class ProgressVisualizerTQDM():
    def __init__(self, total, desc = None, leave = True, ncols = 100, position = 0) -> None:
        self.pbar = tqdm(total = total, desc = desc, leave = leave, ncols = ncols, position = position,bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]{desc}")
    
    def update(self, value):
        self.pbar.update(value)

    def set_description(self, desc):
        self.pbar.set_description(desc)
    
    def close(self):
        self.pbar.close()

    def __iter__(self):
        return self.pbar.__iter__()




import torch
from torch import nn
from mmseg.models.builder import LOSSES
from torch.nn import functional as F
import mmcv
import numpy as np

def get_class_weight(class_weight):
    """Get class weight for loss function.

    Args:
        class_weight (list[float] | str | None): If class_weight is a str,
            take it as a file name and read from it.
    """
    if isinstance(class_weight, str):
        # take it as a file path
        if class_weight.endswith('.npy'):
            class_weight = np.load(class_weight)
        else:
            # pkl, json or yaml
            class_weight = mmcv.load(class_weight)

    return class_weight

def cross_entropy(pred,
                  label,
                  weight=None,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=-100,
                  avg_non_ignore=False):
    """cross_entropy. The wrapper function for :func:`F.cross_entropy`

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
            Default: None.
        class_weight (list[float], optional): The weight for each class.
            Default: None.
        reduction (str, optional): The method used to reduce the loss.
            Options are 'none', 'mean' and 'sum'. Default: 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Default: None.
        ignore_index (int): Specifies a target value that is ignored and
            does not contribute to the input gradients. When
            ``avg_non_ignore `` is ``True``, and the ``reduction`` is
            ``''mean''``, the loss is averaged over non-ignored targets.
            Defaults: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
    """

    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses

    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    # apply weights and do the reduction
    # average loss over non-ignored elements
    # pytorch's official cross_entropy average loss over non-ignored elements
    # refer to https://github.com/pytorch/pytorch/blob/56b43f4fec1f76953f15a627694d4bba34588969/torch/nn/functional.py#L2660  # noqa


    return loss



@LOSSES.register_module()
class PatchWiseCrossEntropyLoss(nn.Module):
    def __init__(self, in_chans=3, patch_size=16, use_sigmoid=False, use_mask=False,
                 reduction='custom', class_weight=None, loss_weight=1.0,
                 loss_name='loss_patch_wise_cross_entropy', avg_non_ignore=False):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.loss = cross_entropy  # Note the 'none' reduction here
        self._loss_name = loss_name
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self.avg_non_ignore = avg_non_ignore

    def forward(self, cls_score, label, weight=None, avg_factor=None,
                reduction_override=None, ignore_index=-100, **kwargs):
        reduction = reduction_override if reduction_override else self.reduction

        
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None

        label = label.unsqueeze(-1)
        label = label.permute(0,3,1,2)
  
        cls_patch = self.patchify(cls_score,24)
        label_patch = self.patchify(label,1)
    

        cls_patch = cls_patch.permute(0,3,1,2)
        label_patch = label_patch.squeeze(-1)

        cls_patch = cls_patch.float()
        label_patch = label_patch.long()

        patch_losses = []
        for i in range(cls_patch.shape[1]):  # Iterate over patches
            patch_loss = self.loss(cls_patch[:, i], label_patch[:, i],
                                   weight, class_weight=class_weight, reduction='none',
                                   avg_factor=avg_factor, avg_non_ignore=self.avg_non_ignore,
                                   ignore_index=ignore_index, **kwargs)
            patch_losses.append(patch_loss.unsqueeze(1))
        patch_losses = torch.cat(patch_losses, dim=1)
        # Custom reduction across patches
        loss = self.custom_reduction(patch_losses)
        return loss

    def custom_reduction(self, patch_losses):
        # Implement your custom reduction logic here
        # Example: weighted mean, max, or any complex logic
        
        # Use torch.max to get the maximum values along the specified axis
        max_values, _ = torch.max(patch_losses, dim=1)  # This unpacks the two tensors returned by torch.max
        
        # Now, max_values is a tensor containing the maximum values
        # You can safely compute the mean of these maximum values
        patch_losses_mean = torch.mean(max_values)
        
        return patch_losses_mean
        


    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
    

    def patchify(self, imgs, channels):
        """
        imgs: (N, C, H, W)
        x: (N, L, patch_size**2 *C)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], channels, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 , channels))
        return x
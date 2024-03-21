# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Reused code from DINOv2 repository https://github.com/facebookresearch/dinov2
# which is licensed under apache License
# found in the paper https://arxiv.org/abs/2304.07193
# References:
#   https://github.com/fundamentalvision/Deformable-DETR/tree/main/models/ops/modules
#   https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0

from .ms_deform_attn import MSDeformAttn

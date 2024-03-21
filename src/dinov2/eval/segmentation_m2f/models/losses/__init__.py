# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Reused code from DINOv2 repository https://github.com/facebookresearch/dinov2
# which is licensed under apache License
# found in the paper https://arxiv.org/abs/2304.07193
from .cross_entropy_loss import CrossEntropyLoss, binary_cross_entropy, cross_entropy, mask_cross_entropy
from .dice_loss import DiceLoss
from .match_costs import ClassificationCost, CrossEntropyLossCost, DiceCost

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Reused code from DINOv2 repository https://github.com/facebookresearch/dinov2
# which is licensed under apache License
# found in the paper https://arxiv.org/abs/2304.07193
from mmcv.utils import Registry

TRANSFORMER = Registry("Transformer")
MASK_ASSIGNERS = Registry("mask_assigner")
MATCH_COST = Registry("match_cost")


def build_match_cost(cfg):
    """Build Match Cost."""
    return MATCH_COST.build(cfg)


def build_assigner(cfg):
    """Build Assigner."""
    return MASK_ASSIGNERS.build(cfg)


def build_transformer(cfg):
    """Build Transformer."""
    return TRANSFORMER.build(cfg)

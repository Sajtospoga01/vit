# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Reused code from DINOv2 repository https://github.com/facebookresearch/dinov2
# which is licensed under apache License
# found in the paper https://arxiv.org/abs/2304.07193

# modified

from src.dinov2.models.base_models import ViT    
from src.hsi_vit.vit_hsi import HSIViT

def build_model_from_cfg(cfg,only_teacher=False):
    return build_model(cfg.student, only_teacher=only_teacher, img_size=cfg.crops.global_crops_size)

def build_model_from_cfg_hsi(cfg,only_teacher=False):
    return build_model(cfg.student, only_teacher=only_teacher, img_size=cfg.crops.global_crops_size,model_class = HSIViT)
    

def build_model(args, only_teacher=False, img_size=64, model_class = ViT):
    args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=args.im_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
        )
        teacher = model_class(**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = model_class(
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim
    return student, teacher, embed_dim
    
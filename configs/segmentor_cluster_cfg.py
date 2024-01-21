num_things_classes = 0
num_stuff_classes = 24
num_classes = 24
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoderMask2Former',
    # pretrained=
    # '/checkpoint/timdarcet/projects/densecluster/23_03_27_bundle_sk2_hrft/eval/training_9999/teacher_checkpoint.mmseg.pth',
    backbone=dict(
        type='ViTAdapter',

        patch_size=8,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        drop_path_rate=0.4,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=12,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        with_cp=True,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        window_attn=[
            False, False, False, False, False, False, False, False, False,
            False, False, False
        ],
        window_size=[
            None, None, None, None, None, None, None, None, None, None, None,
            None,
        ],
        freeze_vit=True,
        use_cls=True,
        pretrain_size=64,
        img_size=64,
        ffn_type='swiglufused'),
    decode_head=dict(
        type='Mask2FormerHead',
        in_channels=[768, 768, 768, 768],
        feat_channels=768,
        out_channels=768,
        in_index=[0, 1, 2, 3],
        num_things_classes=0,
        num_stuff_classes=24,
        num_queries=100,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=768,
                        num_heads=16,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=False,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=768,
                        feedforward_channels=3144,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True),
                        with_cp=True),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                init_cfg=None),
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=384, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=384, normalize=True),
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=9,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=768,
                    num_heads=16,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=768,
                    feedforward_channels=3144,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True,
                    with_cp=True),
                feedforward_channels=3144,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,1.0
            ]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0)),
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type='MaskHungarianAssigner',
            cls_cost=dict(type='ClassificationCost', weight=2.0),
            mask_cost=dict(
                type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
            dice_cost=dict(
                type='DiceCost', weight=5.0, pred_act=True, eps=1.0)),
        sampler=dict(type='MaskPseudoSampler')),
    test_cfg=dict(
        panoptic_on=True,
        semantic_on=False,
        instance_on=True,
        max_per_image=100,
        iou_thr=0.8,
        filter_low_score=True,
        mode='slide',
        crop_size=(512, 512),
        stride=(341, 341)),
    init_cfg=None)
dataset_type = 'WHU_OHS'
data_root = '/nfs/datasets/new_dataset/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile',),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='ToMask'),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_semantic_seg', 'gt_masks', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='WHU_OHS',
        data_root='/nfs/datasets/new_dataset/',
        img_dir='images',
        ann_dir='annotations',
        pipeline=[
            dict(type='MyLoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False), #! possibly set to true
            dict(type='Resize', img_scale=(256, 64), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(64, 64), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            # dict(type='PhotoMetricDistortion'),
            dict(
                type='HSINormalize',
                mean=[  
                    136.43702139, 136.95781982, 136.70735693, 136.91850906, 137.12465157,
                    137.26050865, 137.37743316, 137.24835798, 137.04779119, 136.9453704,
                    136.79646442, 136.68328908, 136.28231996, 136.02395119, 136.01146934,
                    136.72767901, 137.38975674, 137.58604882, 137.61197314, 137.46675538,
                    137.57319831, 137.69239868, 137.72318172, 137.76894864, 137.74861655,
                    137.77535075, 137.80038781, 137.85482571, 137.88595859, 137.9490434,
                    138.00128494, 138.17846624
                    ],
                std=[
                    33.48886853, 33.22482796, 33.4670978, 33.53758141, 33.48675988, 33.33348355,
                    33.35096189, 33.63958817, 33.85081288, 34.08314358, 34.37542553, 34.60344274,
                    34.80732573, 35.17761688, 35.1956623, 34.43121367, 33.76600779, 33.77061146,
                    33.92844916, 34.0370747, 34.0285642, 33.87601205, 33.81035869, 33.66611756,
                    33.74440912, 33.69755911, 33.69845938, 33.6707364, 33.62571536, 33.44615438,
                    33.27907802, 32.90732107
                    ],
                ),
            dict(type='Pad', size=(64, 64), pad_val=0, seg_pad_val=255),
            dict(type='ToMask'),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_semantic_seg','gt_labels' , 'gt_masks']),
         
        ]),
    val=dict(
        type='WHU_OHS',
        data_root='/nfs/datasets/new_dataset/',
        img_dir='f_val/images',
        ann_dir='f_val/labels',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(256, 64),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='ResizeToMultiple', size_divisor=32),
                    dict(type='RandomFlip'),
                    dict(
                        type='HSINormalize',
                        mean=[  
                            136.43702139, 136.95781982, 136.70735693, 136.91850906, 137.12465157,
                            137.26050865, 137.37743316, 137.24835798, 137.04779119, 136.9453704,
                            136.79646442, 136.68328908, 136.28231996, 136.02395119, 136.01146934,
                            136.72767901, 137.38975674, 137.58604882, 137.61197314, 137.46675538,
                            137.57319831, 137.69239868, 137.72318172, 137.76894864, 137.74861655,
                            137.77535075, 137.80038781, 137.85482571, 137.88595859, 137.9490434,
                            138.00128494, 138.17846624
                            ],
                        std=[
                            33.48886853, 33.22482796, 33.4670978, 33.53758141, 33.48675988, 33.33348355,
                            33.35096189, 33.63958817, 33.85081288, 34.08314358, 34.37542553, 34.60344274,
                            34.80732573, 35.17761688, 35.1956623, 34.43121367, 33.76600779, 33.77061146,
                            33.92844916, 34.0370747, 34.0285642, 33.87601205, 33.81035869, 33.66611756,
                            33.74440912, 33.69755911, 33.69845938, 33.6707364, 33.62571536, 33.44615438,
                            33.27907802, 32.90732107
                            ],
                        ),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='WHU_OHS',
        data_root='/nfs/datasets/new_dataset/',
        img_dir='f_ts/images',
        ann_dir='f_ts/labels',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(256, 64),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='ResizeToMultiple', size_divisor=32),
                    dict(type='RandomFlip'),
                    dict(
                        type='HSINormalize',
                        mean=[  
                            136.43702139, 136.95781982, 136.70735693, 136.91850906, 137.12465157,
                            137.26050865, 137.37743316, 137.24835798, 137.04779119, 136.9453704,
                            136.79646442, 136.68328908, 136.28231996, 136.02395119, 136.01146934,
                            136.72767901, 137.38975674, 137.58604882, 137.61197314, 137.46675538,
                            137.57319831, 137.69239868, 137.72318172, 137.76894864, 137.74861655,
                            137.77535075, 137.80038781, 137.85482571, 137.88595859, 137.9490434,
                            138.00128494, 138.17846624
                            ],
                        std=[
                            33.48886853, 33.22482796, 33.4670978, 33.53758141, 33.48675988, 33.33348355,
                            33.35096189, 33.63958817, 33.85081288, 34.08314358, 34.37542553, 34.60344274,
                            34.80732573, 35.17761688, 35.1956623, 34.43121367, 33.76600779, 33.77061146,
                            33.92844916, 34.0370747, 34.0285642, 33.87601205, 33.81035869, 33.66611756,
                            33.74440912, 33.69755911, 33.69845938, 33.6707364, 33.62571536, 33.44615438,
                            33.27907802, 32.90732107
                            ],
                        ),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(
            type='WandbLoggerHook',  # Enables logging to Weights & Biases
            init_kwargs=dict(
                project='vit-dino',  # Name of the W&B project
                # config=dict(your_config_dict),  # Optional: log model configuration
                # ... other `wandb.init()` arguments
            ),

            interval=10,  # Log metrics every 10 iterations
            log_artifact=True,  # If True, log artifacts like checkpoints
            # ... other WandbLoggerHook arguments
        ),
        ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=1.8e-05,
    betas=(0.9, 0.999),
    weight_decay=0.0032,
    constructor='LearningRateDecayOptimizerConstructorVIT',
    paramwise_cfg=dict(num_layers=12, decay_rate=1.0, decay_type='layer_wise')
    )
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=1,out_dir='/nfs/',file_client_args=dict(backend='disk'))
evaluation = dict(
    interval=4000, metric='mIoU', pre_eval=True, save_best='mIoU')
res = 512
# pretrained = '/checkpoint/timdarcet/models/deit_base_patch16_224-b5f2ef4d.pth'
fp16 = False
work_dir = '/nfs/'
gpu_ids = range(0, 16)
auto_resume = True  
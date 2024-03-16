dataset_type = 'WHU_OHS'
data_root = '/nfs/datasets/new_dataset/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (64, 64)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(99999999, 640), ratio_range=(1.0, 3.0)),
    dict(type='RandomCrop', crop_size=(640, 640), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(640, 640), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(99999999, 640),
        img_ratios=[1.0, 1.32, 1.73, 2.28, 3.0],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
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
    samples_per_gpu=16,
    workers_per_gpu=1,
    train=dict(
        type='WHU_OHS',
        data_root='/nfs/datasets/full_data/full/',
        img_dir='images/train',
        ann_dir='annotations/train',
        pipeline=[
            dict(type='MyLoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=True),
            dict(
                type='Resize',
                img_scale=(2048, 512),
                ratio_range=(1.0, 3.0)),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='RepositionData'),
            # dict(type='PhotoMetricDistortion'),
            # dict(
            #     type='HSINormalize',
            #     mean=[  
            #         136.43702139, 136.95781982, 136.70735693, 136.91850906, 137.12465157,
            #         137.26050865, 137.37743316, 137.24835798, 137.04779119, 136.9453704,
            #         136.79646442, 136.68328908, 136.28231996, 136.02395119, 136.01146934,
            #         136.72767901, 137.38975674, 137.58604882, 137.61197314, 137.46675538,
            #         137.57319831, 137.69239868, 137.72318172, 137.76894864, 137.74861655,
            #         137.77535075, 137.80038781, 137.85482571, 137.88595859, 137.9490434,
            #         138.00128494, 138.17846624
            #         ],
            #     std=[
            #         33.48886853, 33.22482796, 33.4670978, 33.53758141, 33.48675988, 33.33348355,
            #         33.35096189, 33.63958817, 33.85081288, 34.08314358, 34.37542553, 34.60344274,
            #         34.80732573, 35.17761688, 35.1956623, 34.43121367, 33.76600779, 33.77061146,
            #         33.92844916, 34.0370747, 34.0285642, 33.87601205, 33.81035869, 33.66611756,
            #         33.74440912, 33.69755911, 33.69845938, 33.6707364, 33.62571536, 33.44615438,
            #         33.27907802, 32.90732107
            #         ],
            #     ),
            dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='WHU_OHS',
        data_root='/nfs/datasets/full_data/full/',
        img_dir='images/validate',
        ann_dir='annotations/validate',
        pipeline=[
            dict(type='MyLoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 512),
                img_ratios=[1.0, 1.32, 1.73, 2.28, 3.0],
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='RepositionData'),
                    # dict(
                    #     type='HSINormalize',
                    #     mean=[  
                    #         136.43702139, 136.95781982, 136.70735693, 136.91850906, 137.12465157,
                    #         137.26050865, 137.37743316, 137.24835798, 137.04779119, 136.9453704,
                    #         136.79646442, 136.68328908, 136.28231996, 136.02395119, 136.01146934,
                    #         136.72767901, 137.38975674, 137.58604882, 137.61197314, 137.46675538,
                    #         137.57319831, 137.69239868, 137.72318172, 137.76894864, 137.74861655,
                    #         137.77535075, 137.80038781, 137.85482571, 137.88595859, 137.9490434,
                    #         138.00128494, 138.17846624
                    #         ],
                    #     std=[
                    #         33.48886853, 33.22482796, 33.4670978, 33.53758141, 33.48675988, 33.33348355,
                    #         33.35096189, 33.63958817, 33.85081288, 34.08314358, 34.37542553, 34.60344274,
                    #         34.80732573, 35.17761688, 35.1956623, 34.43121367, 33.76600779, 33.77061146,
                    #         33.92844916, 34.0370747, 34.0285642, 33.87601205, 33.81035869, 33.66611756,
                    #         33.74440912, 33.69755911, 33.69845938, 33.6707364, 33.62571536, 33.44615438,
                    #         33.27907802, 32.90732107
                    #         ],
                    #     ),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='WHU_OHS',
        data_root='/nfs/datasets/full_data/full/',
        img_dir='images/test',
        ann_dir='annotations/test',
        pipeline=[
            dict(type='MyLoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(99999999, 224),
                img_ratios=[1.0, 1.32, 1.73, 2.28, 3.0],
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[
        dict(type='TextLoggerHook', by_epoch=False, out_dir='/nfs/segmentor/logs'),
        dict(
            type='MMSegWandbHook',  # Enables logging to Weights & Biases
            init_kwargs=dict(
                project='vit-dino',  # Name of the W&B project
                # config=dict(your_config_dict),  # Optional: log model configuration
                # ... other `wandb.init()` arguments
            ),

            interval=10,  # Log metrics every 10 iterations
            # log_artifact=True,  # If True, log artifacts like checkpoints
            log_checkpoint=True,
            log_checkpoint_metadata=True,
            num_eval_images = 10,

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
    type='AdamW', lr=0.0001, weight_decay=0.001, betas=(0.9, 0.999))
optimizer_config = dict(
    type='DistOptimizerHook',
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=False)
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=5000,out_dir='/nfs/segmentor/checkpoints')
evaluation = dict(interval=9999, metric='mIoU', pre_eval=True)
fp16 = None
find_unused_parameters = True
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',

    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        in_channels=32,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.2,
        num_classes=24,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='DiceLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.2,
        num_classes=24,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='DiceLoss', use_sigmoid=False, loss_weight=0.4)),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(32, 32)))

auto_resume = True
gpu_ids = range(0, 8)
work_dir = '/nfs/'
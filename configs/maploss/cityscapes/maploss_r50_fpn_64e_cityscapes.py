_base_ = [
    '../../_base_/datasets/cityscapes_detection.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_1x.py'
]
# work_dir = '/mnt/jys/mmdet3.0/work_dirs/15/'
load_from = 'work_dirs/maploss_r50_fpn_1x_coco/epoch_12.pth'  # noqa
work_dir = 'work_dirs/maploss_r50_fpn_64e_cityscapes'
# model settings
# compile = True
batch_size = 4
model = dict(
    type='SingleStageDetector',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
    ),
    bbox_head=dict(
        type='mAPLossHead',
        num_classes=8,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        with_centerness='mul',
        init_cfg=dict(
            type='Normal',
            layer='Conv2d',
            std=0.01,
            override=[
                dict(type='Normal', name='conv_cls', std=0.0, bias=-4.6),
                dict(type='Normal', name='conv_reg', std=0.0, bias=4.0),
                dict(type='Normal', name='conv_centerness', std=0.0, bias=0),
            ]),
    ),
    # model training and testing settings
    train_cfg=dict(
        loss_weight_mAP=0.01,
        score_th=(-5, 5),
        momentum=0.9,
        discrete_num=200,
        pos_per_gt=10,
        lead_ratio=10,
        iou_weight_alpha=10,
        class_cat='switch',
        evaluator='coco',
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))


# find_unused_parameters = True
# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale=dict(
        init_scale=2 ** 10, growth_factor=2, backoff_factor=0.5, growth_interval=10000,
    ),
    # accumulative_counts=2,
    # type='MyOptimWrapper',
    clip_grad=dict(max_norm=10),
    # paramwise_cfg=dict(norm_decay_mult=0.),
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
    # optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.1, _delete_=True)
)
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=batch_size // 2,
    batch_sampler=dict(drop_last=True),
)
custom_hooks = [
    dict(type='SetEpochHook', ),
]

# val_evaluator = dict(
#     iouwise=True,
#     # classwise=True,
#     # proposal_nums=(1000, 3000, 10000),
# )
# test_evaluator = val_evaluator
# training schedule for 1x
train_cfg = dict(max_epochs=8)


# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=8,
        by_epoch=True,
        milestones=[7, ],
        gamma=0.1)
]
# actual epoch = 8 * 8 = 64

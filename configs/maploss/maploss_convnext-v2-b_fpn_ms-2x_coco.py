_base_ = './maploss_r50_fpn_ms-2x_coco.py'

work_dir = 'work_dirs/maploss_convnext-v2-b_ms-2x_coco'

checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-base_3rdparty-fcmae_in1k_20230104-8a798eaf.pth'  # noqa

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmcls.ConvNeXt',
        arch='base',
        out_indices=[1, 2, 3],
        frozen_stages=1,
        layer_scale_init_value=0.,  # disable layer scale when using GRN
        gap_before_final_norm=False,
        use_grn=True,  # V2 uses GRN
        with_cp=True,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    neck=dict(
        in_channels=[256, 512, 1024],
        start_level=0,
    ),
)
# val_dataloader = dict(
#     batch_size=1,
#     num_workers=1, )
optim_wrapper = dict(
    optimizer=dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0., bypass_duplicate=True))

_base_ = './maploss_r50_fpn_ms-2x_coco.py'


work_dir = 'work_dirs/maploss_r101_fpn_ms-2x_coco'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))


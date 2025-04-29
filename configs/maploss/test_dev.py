_base_ = './maploss_x101-64x4d_fpn_ms_2x_coco.py'


test_dataloader = dict(
    dataset=dict(
        ann_file=_base_.data_root + 'annotations/image_info_test-dev2017.json',
        data_prefix=dict(img='test2017/'),
    ))

test_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    format_only=True,
    ann_file=_base_.data_root + 'annotations/image_info_test-dev2017.json',
    outfile_prefix=_base_.work_dir + '/coco_detection/test')

# mAPLoss
Based on [mmdetection3.0.0](https://github.com/open-mmlab/mmdetection/tree/v3.0.0), After installing mmdetection 3.0.0, add the files I provided and register. We provide config files, log files and checkpoint files in the paper.

| Backbone  | Lr schd |  MS train | val box AP |  test box AP |        Config     |               Download                            |
| :-------: | :-----: | :------: |  :------: | :----------: | :--------------: | 
| R-50-FPN  |   12e   |   N      |     41.8     |   -    |   [config](./configs/maploss/maploss_r50_fpn_1x_coco.py)   |    [model]() \| [log]()  |
| R-50-FPN  |   24e   |   N      |     42.0     |   -    |   [config](./configs/maploss/maploss_r50_fpn_2x_coco.py)   |    [model]() \| [log]()  |
| R-50-FPN  |   24e   |   Y      |     44.9     |   -    |   [config](./configs/maploss/maploss_r50_fpn_ms-2x_coco.py)   |    [model]() \| [log]()  |
| R-101-FPN  |   24e   |   Y      |     46.4     |   -    |   [config](./configs/maploss/maploss_r101_fpn_ms-2x_coco.py)   |    [model]() \| [log]()  |
| X-101-64x4d-FPN  |   24e   |   Y      |     48.0     |   -    |   [config](./configs/maploss/maploss_x101-64x4d_fpn_ms-2x_coco.py)   |    [model]() \| [log]()  |

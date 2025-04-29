# mAPLoss
Based on [mmdetection3.0.0](https://github.com/open-mmlab/mmdetection/tree/v3.0.0), After installing mmdetection 3.0.0, add the files I provided and register. We provide config files, log files and checkpoint files in the paper.
## COCO Dataset
| Backbone  | Lr schd |  MS train | val box AP |  test box AP |  Config  |    Download   |
| :-------: | :-----: | :------: |  :------: | :----------: | :--------------: | :--------: | 
| R-50-FPN  |   12e   |   N      |     41.8     |   -    |   [config](./configs/maploss/maploss_r50_fpn_1x_coco.py)   |    [model](https://1drv.ms/u/c/eac219a4ecb09de4/EZMR-3fYirlHubjUBA-yFbIBYkf5QzqU9CTyInnVER15Ew) \| [log](https://1drv.ms/u/c/eac219a4ecb09de4/Edlg4-qfrGJKta-DGoElOMIBaZhQgS6gIhedPDqlhjHDkQ)  |
| R-50-FPN  |   24e   |   N      |     42.0     |   -    |   [config](./configs/maploss/maploss_r50_fpn_2x_coco.py)   |    [model]() \| [log]()  |
| R-50-FPN  |   24e   |   Y      |     44.9     |   45.3   |   [config](./configs/maploss/maploss_r50_fpn_ms-2x_coco.py)   |    [model]() \| [log]()  |
| R-101-FPN  |   24e   |   Y      |     46.4     |   46.9    |   [config](./configs/maploss/maploss_r101_fpn_ms-2x_coco.py)   |    [model]() \| [log]()  |
| X-101-64x4d-FPN  |   24e   |   Y      |     48.0     |   48.5   |   [config](./configs/maploss/maploss_x101-64x4d_fpn_ms-2x_coco.py)   |    [model]() \| [log]()  |
| R-50-rsb-FPN  |   12e   |   N      |     43.2     |   -    |   [config](./configs/maploss/maploss_r50-rsb-pre_fpn_1x_coco.py)   |    [model]() \| [log]()  |
| ConvNeXt-v2-t-FPN  |   12e   |   N      |     44.0     |   -    |   [config](./configs/maploss/maploss_convnext-v2-t_fpn_1x_coco.py)   |    [model]() \| [log]()  |
| ConvNeXt-v2-b-FPN  |   24e   |   Y      |     48.6     |   -    |   [config](./configs/maploss/maploss_convnext-v2-b_fpn_ms-2x_coco.py)   |    [model]() \| [log]()  |

## CityScapes Dataset
| Backbone  | Lr schd |  MS train | val box AP |   Config  |    Download   |
| :-------: | :-----: | :------: |  :------: | :----------: | :--------------: |
| R-50-FPN  |   64e   |   Y      |     44.7     |   [config](./configs/maploss/cityscapes/maploss_r50_fpn_64e_cityscapes.py)   |    [model](https://drive.google.com/file/d/1i8qCJLXbK1sjz28ki1U1uWV7TEoOjCPZ/view?usp=sharing) \| [log](https://drive.google.com/file/d/13hhPudIE8IfdpY_LvQAECw_NFgGwmToO/view?usp=sharing)  |

## VOC Dataset
| Backbone  | Lr schd |  MS train | box AP (voc style) |   Config  |    Download   |
| :-------: | :-----: | :------: |  :------: | :----------: | :--------------: |
| R-50-FPN  |   12e   |   N      |     79.1     |   [config](./configs/maploss/voc/maploss_r50_fpn_1x_voc.py)   |    [model]() \| [log]()  |

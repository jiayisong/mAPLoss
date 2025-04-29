# mAPLoss
Based on [mmdetection3.0.0](https://github.com/open-mmlab/mmdetection/tree/v3.0.0), After installing mmdetection 3.0.0, add the files I provided and register. We provide config files, log files and checkpoint files in the paper.
## COCO Dataset
| Backbone  | Lr schd |  MS train | val box AP |  test box AP |  Config  |    Download   |
| :-------: | :-----: | :------: |  :------: | :----------: | :--------------: | :--------: | 
| R-50-FPN  |   12e   |   N      |     41.8     |   -    |   [config](./configs/maploss/maploss_r50_fpn_1x_coco.py)   |    [model](https://1drv.ms/u/c/eac219a4ecb09de4/EZMR-3fYirlHubjUBA-yFbIBYkf5QzqU9CTyInnVER15Ew) \| [log](https://1drv.ms/u/c/eac219a4ecb09de4/Edlg4-qfrGJKta-DGoElOMIBaZhQgS6gIhedPDqlhjHDkQ)  |
| R-50-FPN  |   24e   |   N      |     42.0     |   -    |   [config](./configs/maploss/maploss_r50_fpn_2x_coco.py)   |    [model](https://1drv.ms/u/c/eac219a4ecb09de4/EcmUE9bAWZ5MuKS5htgUpr4BnnmAWopCXEvVeaRJlx1MSg) \| [log](https://1drv.ms/u/c/eac219a4ecb09de4/EX3o9qcR4ExAhCKUUdJRklsBx4CRy8C6ZH69B5vt7_HpgQ?e=myLwy2)  |
| R-50-FPN  |   24e   |   Y      |     44.9     |   45.3   |   [config](./configs/maploss/maploss_r50_fpn_ms-2x_coco.py)   |    [model](https://1drv.ms/u/c/eac219a4ecb09de4/EdCei5IiBuZAixEFqGVGdYkBvF1govZqnZJsPoCopQd01w?e=99I3UR) \| [log](https://1drv.ms/u/c/eac219a4ecb09de4/EfkbU2NyQVZDqhZA4-FAY1QBhGF-oPkWeyvItjP8nqLBAA?e=699cKI)  |
| R-101-FPN  |   24e   |   Y      |     46.4     |   46.9    |   [config](./configs/maploss/maploss_r101_fpn_ms-2x_coco.py)   |    [model](https://1drv.ms/u/c/eac219a4ecb09de4/EXmXUYaLSPVBsjr5I60G02ABbPP5g8NuU2wnWJDsTrJvbg?e=fofRfg) \| [log](https://1drv.ms/u/c/eac219a4ecb09de4/ESSTQndvj_5PrXCPV9lLqssB7S2NoECt3lRf98O_766l8A?e=cNVASz)  |
| X-101-64x4d-FPN  |   24e   |   Y      |     48.0     |   48.5   |   [config](./configs/maploss/maploss_x101-64x4d_fpn_ms-2x_coco.py)   |    [model](https://1drv.ms/u/c/eac219a4ecb09de4/ERGvJQhq3DNFpqYvw0rSUPwBEcLQXgc6lt0iZWrGHjes4Q?e=dEchYL) \| [log](https://1drv.ms/u/c/eac219a4ecb09de4/EdilwyOF1ghIiCqapwcViZ4BWVK5g-R2CEjs6WF6I162Rw?e=fPyXLr)  |
| R-50-rsb-FPN  |   12e   |   N      |     43.2     |   -    |   [config](./configs/maploss/maploss_r50-rsb-pre_fpn_1x_coco.py)   |    [model](https://1drv.ms/u/c/eac219a4ecb09de4/Eb82oY2dtF9Gv-BUk499SqUBBMSwVjwv7AtzWFKetUFTlA?e=7bEUo0) \| [log](https://1drv.ms/u/c/eac219a4ecb09de4/ESV36sVKJhBJguotlMwK3YkBi8k8P9da0WD1IZNHfWveNQ?e=wEdojn)  |
| ConvNeXt-v2-t-FPN  |   12e   |   N      |     44.0     |   -    |   [config](./configs/maploss/maploss_convnext-v2-t_fpn_1x_coco.py)   |    [model](https://1drv.ms/u/c/eac219a4ecb09de4/Ea4RKefvFvNIqpQR5d3VNJEB-USznfdeXU5hmlwfOzpKZw?e=eNRyBd) \| [log](https://1drv.ms/u/c/eac219a4ecb09de4/Edetq_i7TQlEvxYuCWqn8UUBn7JKE1HU5JKbmFiZFkL3Bw?e=4maaAh)  |
| ConvNeXt-v2-b-FPN  |   24e   |   Y      |     48.6     |   -    |   [config](./configs/maploss/maploss_convnext-v2-b_fpn_ms-2x_coco.py)   |    [model](https://1drv.ms/u/c/eac219a4ecb09de4/Ed7tQA2MSSlNjk53YXWFb2gBhHYdvD6Gc_Bk_k0Hg_G6rQ?e=zAAbB7) \| [log](https://1drv.ms/u/c/eac219a4ecb09de4/ERuvkZKiW5BMnZqMXQnwVKIB6Sd5wUqdl6A4oGFmOwymIg?e=BmDfno)  |

## CityScapes Dataset
| Backbone  | Lr schd |  MS train | val box AP |   Config  |    Download   |
| :-------: | :-----: | :------: |  :------: | :----------: | :--------------: |
| R-50-FPN  |   64e   |   Y      |     44.7     |   [config](./configs/maploss/cityscapes/maploss_r50_fpn_64e_cityscapes.py)   |    [model](https://drive.google.com/file/d/1i8qCJLXbK1sjz28ki1U1uWV7TEoOjCPZ/view?usp=sharing) \| [log](https://drive.google.com/file/d/13hhPudIE8IfdpY_LvQAECw_NFgGwmToO/view?usp=sharing)  |

## VOC Dataset
| Backbone  | Lr schd |  MS train | box AP (voc style) |   Config  |    Download   |
| :-------: | :-----: | :------: |  :------: | :----------: | :--------------: |
| R-50-FPN  |   12e   |   N      |     79.1     |   [config](./configs/maploss/voc/maploss_r50_fpn_1x_voc.py)   |    [model](https://1drv.ms/u/c/eac219a4ecb09de4/EcbkYDP6R3NCnuHe4NIu2OYBVTKuJgRHsoYXew6ZjAKm6w?e=QADpad) \| [log](https://1drv.ms/u/c/eac219a4ecb09de4/ER2usE5vAm1Hpd_5ivEAGaEBnkami9Z6Jfc5TakN-YvQXg?e=XwpqKW)  |

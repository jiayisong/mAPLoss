# Copyright (c) OpenMMLab. All rights reserved.
import csv
import math
import os
import time
from typing import Any, List, Sequence, Tuple, Union, Optional

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from numpy import ndarray
from torch import Tensor
from mmengine.config import ConfigDict
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.utils import (ConfigType, InstanceList, MultiConfig, OptConfigType,
                         OptInstanceList)
from ..task_modules.prior_generators import MlvlPointGenerator
from ..utils import multi_apply
from .base_dense_head import BaseDenseHead
from mmengine.structures import InstanceData
from mmengine.model.weight_init import bias_init_with_prob
from mmengine.visualization import Visualizer
from mmengine.logging import print_log, MMLogger
import torch.distributed as dist
from mmcv.ops import batched_nms
from mmdet.structures.bbox import (cat_boxes, get_box_tensor, get_box_wh,
                                   scale_boxes)
from mmcv.cnn import Scale
import torch.nn.functional as F

StrideType = Union[Sequence[int], Sequence[Tuple[int, int]]]

FACTOR = 1e-3


@MODELS.register_module()
class mAPLossHead(BaseDenseHead):
    """Anchor-free head (FCOS, Fovea, RepPoints, etc.).

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Used in child classes.
        stacked_convs (int): Number of stacking convs of the head.
        strides (Sequence[int] or Sequence[Tuple[int, int]]): Downsample
            factor of each feature map.
        dcn_on_last_conv (bool): If true, use dcn in the last layer of
            towers. Defaults to False.
        conv_bias (bool or str): If specified as `auto`, it will be decided by
            the norm_cfg. Bias of conv will be set as True if `norm_cfg` is
            None, otherwise False. Default: "auto".
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        conv_cfg (:obj:`ConfigDict` or dict, Optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, Optional): Config dict for
            normalization layer. Defaults to None.
        train_cfg (:obj:`ConfigDict` or dict, Optional): Training config of
            anchor-free head.
        test_cfg (:obj:`ConfigDict` or dict, Optional): Testing config of
            anchor-free head.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.
    """  # noqa: W605

    _version = 1

    def __init__(
            self,
            num_classes: int,
            in_channels: int,
            feat_channels: int = 256,
            stacked_convs: int = 4,
            strides: StrideType = (4, 8, 16, 32, 64),
            with_centerness='mul',
            dcn_on_last_conv: bool = False,
            conv_bias: Union[bool, str] = 'auto',
            conv_cfg: OptConfigType = None,
            norm_cfg: OptConfigType = dict(
                type='GN', num_groups=32, requires_grad=True),
            train_cfg: OptConfigType = None,
            test_cfg: OptConfigType = None,
            init_cfg: MultiConfig = dict(
                type='Normal',
                layer='Conv2d',
                std=0.01,
                override=[
                    dict(type='Normal', name='conv_cls', std=0.0, bias=-4.6),
                    dict(type='Normal', name='conv_reg', std=0.0, bias=4.0),
                ]),
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.with_centerness = with_centerness
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.prior_generator = MlvlPointGenerator(strides)
        # train_cfg.ds = (train_cfg.score_max - train_cfg.score_min) / train_cfg.discrete_num
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

        s = torch.arange(0.0, train_cfg.discrete_num, )
        s = s / (self.train_cfg.discrete_num - 1) * (
                self.train_cfg.score_th[0] - self.train_cfg.score_th[1]) + self.train_cfg.score_th[1]
        self.register_buffer('s_benckmarch', s)

        iou = 1 - torch.arange(0.0, train_cfg.discrete_num) / (train_cfg.discrete_num - 1)
        self.register_buffer('iou_benckmarch', iou)

        iou_weight = torch.sigmoid(0 * (iou - 0.5))
        self.register_buffer('iou_weight', iou_weight.view(1, 1, -1))

        xy_discrete = torch.stack(torch.meshgrid(s, torch.logit(iou, 1e-7)), dim=-1)
        self.register_buffer('s_iou_benckmarch', xy_discrete)  # [discrete_num, discrete_num, 2]

        self.register_buffer('iou_trans_weight', 1 / (iou * (1 - iou)).clamp_min(1e-7))  # [discrete_num, ]

        self.register_buffer('pos_iou_mean', torch.ones([num_classes, 2]) * 0)
        self.register_buffer('pos_iou_var', torch.ones([num_classes, 2]) * 1)
        self.register_buffer('pos_score_n', torch.ones([num_classes, ]) * 1000)
        self.register_buffer('mAP_matrix',
                             torch.zeros([num_classes, train_cfg.discrete_num, train_cfg.discrete_num, ],
                                         dtype=torch.float32))

        self.epoch = 0
        self.ds = (self.train_cfg.score_th[1] - self.train_cfg.score_th[0]) / (self.train_cfg.discrete_num - 1)
        self.di = 1 / (self.train_cfg.discrete_num - 1)
        # self.get_LA_mat(self.get_tb_gb_N())
        self._init_layers()

    def forward_single(self, x: Tensor
                       , scale: Scale
                       ) -> Tuple[Tensor, ...]:
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.

        Returns:
            tuple: Scores for each class, bbox predictions, features
            after classification and regression conv layers, some
            models needs these features like FCOS.
        """
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.conv_cls(cls_feat).float()

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)
        bbox_pred = scale(bbox_pred).float()
        bbox_pred = bbox_pred.clamp(min=0)
        if self.with_centerness == 'avg':
            centerness = self.conv_centerness(reg_feat).float()
            cls_score = 0.5 * cls_score + 0.5 * centerness
        elif self.with_centerness == 'mul':
            centerness = self.conv_centerness(reg_feat).float()
            cls_score = ScoreCenternessFunction.apply(cls_score, centerness)
        else:
            cls_score = cls_score
            # cls_score = -torch.log(torch.exp(-cls_score) + torch.exp(-centerness) + torch.exp(-centerness - cls_score))
        # cls_score_exp = cls_score.exp()
        # centerness_exp = centerness.exp()
        # cls_score = torch.log((cls_score_exp * centerness_exp) / (cls_score_exp + centerness_exp + 1) + 1e-7)
        # cls_score = 0.5 * cls_score + 0.5 * centerness

        return cls_score, bbox_pred, cls_feat, reg_feat

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        """

        assert len(cls_scores) == len(bbox_preds)
        flatten_cls_scores, flatten_bbox, flatten_points, all_level_points = self.decode_bbox(cls_scores, bbox_preds)

        batch_gt_bbox, batch_gt_label, batch_gt_flag = self.get_target(batch_gt_instances)
        # LA
        # torch.cuda.synchronize()
        # start_time = time.time()
        loss1, pos_score, iou_with_gt, pos_label, back_score, back_label, class_gt_num = self.remove_background(
            flatten_cls_scores, flatten_bbox, flatten_points, all_level_points, batch_gt_bbox, batch_gt_label,
            batch_gt_flag)
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print('LA', end_time - start_time)
        # compute loss
        pos_score_norm = self.score2index(pos_score)
        back_score_norm = self.score2index(back_score)
        iou_norm = self.iou2index(iou_with_gt)
        t1 = self.Pos2Distribution(pos_score_norm, iou_norm, pos_label)
        t2 = self.Neg2Distribution(back_score_norm, back_label)

        t1, t2, batch_gt_flag = self.gather_dist_Ngt(t1, t2, batch_gt_flag.sum() * FACTOR)
        t1_fake = self.get_tb_gb_N()
        mAP = self.get_mAP(t1, t2, t1_fake)
        self.get_LA_mat(t1_fake)

        if dist.is_initialized():
            N = dist.get_world_size()
        else:
            N = 1
        # torch.cuda.synchronize()
        # print('loss', time.time() - end_time)
        return dict(
            # pos_score_mean=pos_score.detach().mean(),
            # neg_score_mean=back_score.detach().mean(),
            # mAP=mAP,
            # dy_k=class_gt_num.sum() / batch_gt_flag,
            # avg_IOU=iou_with_gt.mean().detach(),
            loss_mAP=-mAP * N * self.train_cfg.loss_weight_mAP * (2 * self.train_cfg.lead_ratio + 1),
        )

    # @profile
    def remove_background(self, flatten_cls_scores, flatten_bbox, flatten_points, all_level_points, batch_bbox,
                          batch_label, pad_flag):
        # [b, c, n], [b, 4, n], [1, 4, n], listof[n, 4], [b, n_gt, 4], [b, n_gt], [b, n_gt]
        b, c, n = flatten_cls_scores.shape
        _, n_gt = batch_label.shape
        x = flatten_points[:, 0:1, :]
        y = flatten_points[:, 1:2, :]

        in_gt_bbox = (x > batch_bbox[:, :, 0:1]) * (x < batch_bbox[:, :, 2:3]) * (y > batch_bbox[:, :, 1:2]) * (
                y < batch_bbox[:, :, 3:4])  # [b, n_gt, n]
        in_gt_mask = pad_flag.unsqueeze(-1)  # [b, n_gt, 1]

        gt_center = 0.5 * (batch_bbox[:, :, 2:] + batch_bbox[:, :, :2])  # [b, n_gt, 2]
        gt_wh = batch_bbox[:, :, 2:] - batch_bbox[:, :, :2]  # [b, n_gt, 2]
        wh_ratio = gt_wh[:, :, 0:1] / gt_wh[:, :, 1:2].clamp_min(1e-4)  # [b, n_gt, 1]
        # wh_ratio = torch.ones_like(wh_ratio)
        h = torch.sqrt(self.train_cfg.pos_per_gt / math.pi / wh_ratio)  # [b, n_gt, 1]
        w = torch.sqrt(self.train_cfg.pos_per_gt / math.pi * wh_ratio)  # [b, n_gt, 1]
        wh = torch.cat([w, h], dim=2).unsqueeze(-1) * flatten_points[:, None, 2:, :]  # [b, n_gt, 2, n]

        # wh = gt_wh.unsqueeze(-1) * (math.sqrt(self.train_cfg.pos_per_gt / math.pi) / 8) # [b, n_gt, 2, 1]

        center_xy_dist = flatten_points[:, None, :2, :] - gt_center.unsqueeze(-1)  # [b, n_gt, 2, n]

        center_dists = torch.norm(center_xy_dist / wh, dim=2, keepdim=False)  # [b, n_gt, n]

        # in_center = []
        # center_dists = torch.split(center_dists, [i.shape[0] for i in all_level_points], dim=2)
        # for cd in center_dists:
        #     _, ind = torch.topk(cd, self.train_cfg.pos_per_gt, dim=2, largest=False)
        #     ic = torch.zeros_like(cd, dtype=torch.bool)
        #     ic.scatter_(dim=2, index=ind, src=torch.ones_like(ind, dtype=torch.bool))
        #     in_center.append(ic)
        # in_center = torch.cat(in_center, 2)

        in_center = center_dists <= 1
        # in_center = 1 - (1 - torch.exp(-center_dists.pow(2))).pow(4)
        with torch.no_grad():
            ious = self.compute_iou(batch_bbox, flatten_bbox.transpose(1, 2))
            batch_label_expand = batch_label.unsqueeze(-1).expand([b, n_gt, n])
            scores = torch.gather(flatten_cls_scores, dim=1, index=batch_label_expand)  # [b, n_gt, n]
            candidate_bag = in_gt_bbox * in_center * in_gt_mask  # * in_fpn_level
            mAP = self.mAP_matrix
            mAP = self.de_gather(mAP, batch_label_expand, scores, ious)  # [b, n_gt, n]
            # print(mAP.min(), mAP.max())
            cost_matrix = mAP * candidate_bag
            # cost_matrix = ious * candidate_bag
            topk_value, topk_index = torch.topk(cost_matrix, dim=2, k=self.train_cfg.pos_per_gt)  # [b,  n_gt, 10]
            valid = topk_value > 0
            topk_index2 = torch.where(valid, topk_index, -1)
            index_temp = topk_index2 + topk_value.clamp_min(0)

            _, sort_index = torch.sort(index_temp.view(b, -1), dim=1)
            index_temp = torch.gather(topk_index2.view(b, -1), dim=1, index=sort_index)
            index_temp = (torch.diff(index_temp, dim=1, append=-index_temp.new_ones([b, 1])) != 0)
            pred_mask = torch.zeros_like(index_temp, dtype=torch.bool)
            pred_mask.scatter_(dim=1, index=sort_index, src=index_temp)
            pred_mask = pred_mask.view(b, n_gt, self.train_cfg.pos_per_gt) * valid  # [b,  n_gt, 10]
            score_index = batch_label.unsqueeze(-1) * n + topk_index  # [b,  n_gt, 10]
            batch_score_index = torch.arange(0, b, device=score_index.device, dtype=torch.long).view(b, 1,
                                                                                                     1) * c * n + score_index

            # pred_mask = pred_mask * valid
        pos_bbox = torch.gather(flatten_bbox, dim=2, index=topk_index.view(b, 1, -1). \
                                expand(b, 4, n_gt * self.train_cfg.pos_per_gt)). \
            view(b, 4, n_gt, self.train_cfg.pos_per_gt).permute(0, 2, 3, 1)  # [b, n_gt, 10, 4]
        pos_score = torch.gather(flatten_cls_scores.view(b, -1), dim=1, index=score_index.view(b, -1)). \
            view(b, n_gt, self.train_cfg.pos_per_gt)  # [b, n_gt, 10]

        pos_label = batch_label.unsqueeze(-1).expand_as(pos_score)
        iou_with_gt = self.compute_iou(pos_bbox, batch_bbox.unsqueeze(2))  # [b, n_gt, 10]
        iou_with_gt = iou_with_gt.squeeze(-1)

        # pred_mask, pred_mask2 = self.train_nms(pos_score, iou_with_gt, pred_mask)
        pred_mask2 = pred_mask
        back_mask = flatten_cls_scores.new_zeros([b * c * n], dtype=torch.bool)
        back_mask.scatter_add_(dim=0, index=batch_score_index.view(-1), src=pred_mask2.view(-1))
        back_mask = ~back_mask

        loss = 0
        pred_mask = pred_mask.view(-1)
        pos_score = pos_score.view(-1)
        iou_with_gt = iou_with_gt.view(-1)
        # iou_with_gt2 = iou_with_gt2.view(-1)
        pos_label = pos_label.reshape(-1)
        pos_score = pos_score[pred_mask]
        iou_with_gt = iou_with_gt[pred_mask]
        pos_label = pos_label[pred_mask]
        # iou_with_gt2 = iou_with_gt2[pred_mask]
        back_mask = back_mask.view(b, c, n)
        back_batch, back_label, back_index = torch.nonzero(back_mask, as_tuple=True)
        back_score = flatten_cls_scores[back_batch, back_label, back_index]

        # back_mask = torch.split(back_mask, [i.shape[0] for i in all_level_points], dim=2)
        # back_score = torch.split(flatten_cls_scores, [i.shape[0] for i in all_level_points], dim=2)
        # back_s = []
        # back_l = []
        # for m, s in zip(back_mask, back_score):
        #     pred_num = s.shape[2]
        #     s, ind = torch.topk(s.reshape(b, -1), 1000, dim=1)
        #     m = m.reshape(b, -1)
        #     m = torch.gather(m, 1, ind)
        #     back_label = ind // pred_num
        #     back_label = back_label[m]
        #     s = s[m]
        #     back_s.append(s)
        #     back_l.append(back_label)
        # back_score = torch.cat(back_s, 0)
        # back_label = torch.cat(back_l, 0)

        with torch.no_grad():
            pos_iou = torch.stack([pos_score, torch.logit(iou_with_gt, eps=1e-7)], 1)
            pos_iou_sum = flatten_cls_scores.new_zeros([self.num_classes, 2])
            pos_iou_square_sum = flatten_cls_scores.new_zeros([self.num_classes, 2])
            pos_iou_sum.index_add_(0, pos_label, pos_iou)
            pos_iou_square_sum.index_add_(0, pos_label, pos_iou.square())
            pos_score_n = flatten_cls_scores.new_zeros([self.num_classes, ])
            pos_score_n.scatter_add_(0, pos_label, torch.ones_like(pos_score))
            # score_iou_sum = flatten_cls_scores.new_zeros([self.num_classes, ])
            # score_iou_sum.scatter_add_(0, pos_label, pos_iou[:, 0] * pos_iou[:, 1])
            #
            # score_sum = flatten_cls_scores.sum(dim=[0, 2], keepdims=False)
            # score_square_sum = flatten_cls_scores.square().sum(dim=[0, 2], keepdims=False)
            # neg_score_n = back_mask.sum(dim=[0, 2], keepdims=False)
            # neg_score_sum = score_sum - pos_iou_sum[:, 0]
            # neg_square_score_sum = score_square_sum - pos_iou_square_sum[:, 0]

            if dist.is_initialized():
                dist.all_reduce(pos_iou_sum)
                dist.all_reduce(pos_iou_square_sum)
                dist.all_reduce(pos_score_n)
                # dist.all_reduce(score_iou_sum)
                # dist.all_reduce(neg_score_sum)
                # dist.all_reduce(neg_square_score_sum)
                # dist.all_reduce(neg_score_n)
            pos_iou_mean = pos_iou_sum / pos_score_n.unsqueeze(1)
            pos_iou_var = (pos_iou_square_sum / pos_score_n.unsqueeze(1) - pos_iou_mean.square())

            # cov = score_iou_sum / pos_score_n - pos_iou_mean[:, 0] * pos_iou_mean[:, 1]
            #
            # neg_score_mean = neg_score_sum / neg_score_n
            # neg_score_var = (neg_square_score_sum / neg_score_n - neg_score_mean.square())

            # pos_iou_mean = pos_iou_mean.clamp_min(-3)
            # neg_score_mean = neg_score_mean.clamp_max(-5)
            mask1 = pos_score_n > 0
            pos_iou_mean = torch.where(mask1.unsqueeze(1).expand_as(pos_iou_mean), pos_iou_mean, self.pos_iou_mean)
            pos_iou_var = torch.where(mask1.unsqueeze(1).expand_as(pos_iou_mean), pos_iou_var, self.pos_iou_var)
            pos_score_n2 = torch.where(mask1, pos_score_n, self.pos_score_n)
            # cov = torch.where(mask1, cov, self.cov)
            # mask2 = neg_score_n > 0
            # neg_score_mean = torch.where(mask2, neg_score_mean, self.neg_score_mean)
            # neg_score_var = torch.where(mask2, neg_score_var, self.neg_score_var)
            self.pos_iou_mean = self.pos_iou_mean * self.train_cfg.momentum + pos_iou_mean * (
                    1 - self.train_cfg.momentum)
            self.pos_iou_var = self.pos_iou_var * self.train_cfg.momentum + pos_iou_var * (
                    1 - self.train_cfg.momentum)
            self.pos_score_n = self.pos_score_n * self.train_cfg.momentum + pos_score_n2 * (1 - self.train_cfg.momentum)
            # self.neg_score_mean = self.neg_score_mean * self.train_cfg.momentum + neg_score_mean * (
            #         1 - self.train_cfg.momentum)
            # self.neg_score_var = self.neg_score_var * self.train_cfg.momentum + neg_score_var * (
            #         1 - self.train_cfg.momentum)
            # self.neg_score_n = self.neg_score_n * self.train_cfg.momentum + neg_score_n * (1 - self.train_cfg.momentum)
            # self.cov = self.cov * self.train_cfg.momentum + cov * (1 - self.train_cfg.momentum)
            # iou_th = torch.sigmoid(self.pos_iou_mean[:, 1] - self.pos_iou_var[:, 1].sqrt() * 3).clamp_max(0.5).view(-1,
            #                                                                                                         1,
            #                                                                                                         1)
            # self.iou_weight = (self.iou_benckmarch >= iou_th)
            # print(torch.sigmoid(self.pos_iou_mean[:, 1] - self.pos_iou_var[:, 1].sqrt() * 3))
            # z = - self.pos_iou_mean[:, 1] / (self.pos_iou_var[:, 1].sqrt() * math.sqrt(2))
            # # 计算 CDF
            # cdf = 0.5 * (1 + torch.erf(z))
            # a = torch.tan(math.pi * (0.5 - cdf)) / 0.22
            # print(cdf)
            # self.iou_weight = torch.sigmoid(a.view(-1, 1, 1) * (self.iou_benckmarch - 0.5))
            # mean = torch.sigmoid((pos_iou_mean[:, 1] * pos_score_n).sum() / pos_score_n.sum().clamp_min(1e-7))
            # a = -self.train_cfg.iou_weight_alpha * torch.log(1 - mean)
            # a = -self.train_cfg.iou_weight_alpha * torch.log(1 - torch.sigmoid(pos_iou_mean[:, 1])).view(-1, 1, 1)
            # self.iou_weight = torch.sigmoid(a * (self.iou_benckmarch - 0.5))
            # print(torch.sigmoid(mean))
        return loss, pos_score, iou_with_gt, pos_label, back_score, back_label, pos_score_n * FACTOR

    def score2index(self, score):
        norm_score = (score - self.train_cfg.score_th[1]) / (
                self.train_cfg.score_th[0] - self.train_cfg.score_th[1]) * (self.train_cfg.discrete_num - 1)
        return norm_score

    def iou2index(self, flatten_ious):
        flatten_ious = (1 - flatten_ious) * (self.train_cfg.discrete_num - 1)
        return flatten_ious

    def de_gather(self, mAP_mat, label, norm_score, norm_iou):
        norm_score = self.score2index(norm_score).clamp(1e-4, self.train_cfg.discrete_num - 1 - 1e-4)
        norm_iou = self.iou2index(norm_iou).clamp(1e-4, self.train_cfg.discrete_num - 1 - 1e-4)
        score_min_ind = torch.floor(norm_score).long()
        score_max_ind = score_min_ind + 1
        score_max_val = norm_score - score_min_ind
        score_min_val = 1 - score_max_val

        iou_min_ind = torch.floor(norm_iou).long()
        iou_max_ind = iou_min_ind + 1
        iou_max_val = norm_iou - iou_min_ind
        iou_min_val = 1 - iou_max_val

        f11 = mAP_mat[label, score_min_ind, iou_min_ind]
        f12 = mAP_mat[label, score_min_ind, iou_max_ind]
        f21 = mAP_mat[label, score_max_ind, iou_min_ind]
        f22 = mAP_mat[label, score_max_ind, iou_max_ind]

        mAP = f11 * iou_min_val * score_min_val + f12 * score_min_val * iou_max_val + f21 * score_max_val * iou_min_val + f22 * score_max_val * iou_max_val
        return mAP

    # def de_gather(self, mAP_mat, label, norm_score, norm_iou):
    #     norm_score = self.score2index(norm_score).long().clamp(0, self.train_cfg.discrete_num - 1)
    #     norm_iou = self.iou2index(norm_iou).long().clamp(0, self.train_cfg.discrete_num - 1)
    #     return mAP_mat[label, norm_score, norm_iou]

    def todist(self, norm_score, label, norm_iou=None):
        score_min_ind = torch.floor(norm_score)
        score_max_ind = score_min_ind + 1
        score_max_val = norm_score - score_min_ind
        score_min_val = 1 - score_max_val
        if norm_iou is not None:
            iou_min_ind = torch.floor(norm_iou)
            iou_max_ind = iou_min_ind + 1
            iou_max_val = norm_iou - iou_min_ind
            iou_min_val = 1 - iou_max_val

            score_val = torch.stack([score_min_val, score_max_val], dim=1).view(-1, 2, 1)
            score_ind = torch.stack([score_min_ind, score_max_ind], dim=1).long().view(-1, 2, 1)

            iou_val = torch.stack([iou_min_val, iou_max_val], dim=1).view(-1, 1, 2)
            iou_ind = torch.stack([iou_min_ind, iou_max_ind], dim=1).long().view(-1, 1, 2)

            val = score_val * iou_val * FACTOR
            ind = label.view(-1, 1,
                             1) * self.train_cfg.discrete_num * (self.train_cfg.discrete_num) + score_ind * (
                      self.train_cfg.discrete_num) + iou_ind

            distribution1 = norm_score.new_zeros(
                [self.num_classes * self.train_cfg.discrete_num * (self.train_cfg.discrete_num), ])
            distribution1.scatter_add_(0, ind.view(-1), val.view(-1))
            distribution1 = distribution1.view(self.num_classes, self.train_cfg.discrete_num,
                                               self.train_cfg.discrete_num)
            return distribution1
            # return (distribution1[:, :, 0] + distribution1[:, :, 1]).contiguous(), distribution1[:, :, 0].contiguous()
        else:
            score_val = torch.stack([score_min_val, score_max_val], dim=1) * FACTOR  # * w.unsqueeze(1)
            score_ind = torch.stack([score_min_ind, score_max_ind], dim=1).long()
            score_ind = score_ind + label.unsqueeze(-1) * self.train_cfg.discrete_num
            score_ind = score_ind.view(-1)
            distribution1 = norm_score.new_zeros([self.num_classes * self.train_cfg.discrete_num, ])
            distribution1.scatter_add_(0, score_ind, score_val.view(-1))
            distribution1 = distribution1.view(self.num_classes, self.train_cfg.discrete_num)
            return distribution1

    # @profile
    def Pos2Distribution(self, norm_score, flatten_ious, label):
        norm_score = ClampGradientFunction.apply(norm_score, 1e-4, self.train_cfg.discrete_num - 1 - 1e-4)
        flatten_ious = ClampGradientFunction.apply(flatten_ious, 1e-4, self.train_cfg.discrete_num - 1 - 1e-4)
        distribution1 = self.todist(norm_score, label, flatten_ious)
        return distribution1

    # @profile
    def Neg2Distribution(self, norm_score, label):
        lb = 1e-4
        hb = self.train_cfg.discrete_num - 1 - 1e-4
        mask = (norm_score < hb)
        norm_score = norm_score[mask]
        label = label[mask]
        norm_score = torch.where(norm_score <= lb, norm_score - norm_score.detach() + lb, norm_score)
        distribution = self.todist(norm_score, label)
        return distribution

    def set_epoch(self, epoch, max_epochs):
        print_log(f'set epoch {epoch} / {max_epochs}', logger=MMLogger.get_current_instance())
        self.epoch = epoch / (max_epochs - 1)
        a = -self.train_cfg.iou_weight_alpha * (math.log(max(1 - self.epoch, 1e-7)))
        if self.train_cfg.evaluator == 'coco':
            iou_weight = torch.sigmoid(a * (self.iou_benckmarch - 0.5))
        elif self.train_cfg.evaluator == 'voc':
            iou_weight = torch.sigmoid(a * (self.iou_benckmarch - 0.5))
            iou_weight = iou_weight * (1 - iou_weight)
        else:
            raise RuntimeError('train_cfg.evaluator error')
        self.iou_weight = iou_weight.view(1, 1, -1)


    def get_tb_gb_N(self):
        pos_iou_mean = self.pos_iou_mean.clone()
        pos_iou_var = self.pos_iou_var.clone().clamp_min(0.1)

        pos_iou_mean0 = torch.stack([pos_iou_mean[:, 0], pos_iou_mean[:, 0]], dim=1)
        pos_iou_mean1 = torch.stack([pos_iou_mean[:, 1], pos_iou_mean[:, 1]], dim=1)
        pos_iou_var0 = torch.stack([pos_iou_var[:, 0], pos_iou_var[:, 0]], dim=1)
        pos_iou_var1 = torch.stack([pos_iou_var[:, 1], pos_iou_var[:, 1]], dim=1)
        cov0 = pos_iou_var[:, 0] * 0.99
        cov1 = pos_iou_var[:, 1] * 0.99
        t1_fake = self.gaussian_pdf_2d(pos_iou_mean1, pos_iou_var1, cov1, self.pos_score_n)
        t1_fake = t1_fake + self.gaussian_pdf_2d(pos_iou_mean0, pos_iou_var0, cov0, self.pos_score_n)  # [C, N_s]

        # pos_iou_mean0 = torch.stack([pos_iou_mean[:, 0], pos_iou_mean[:, 0]], dim=1)
        # pos_iou_mean1 = torch.stack([pos_iou_mean[:, 1], pos_iou_mean[:, 1]], dim=1)
        # t1_fake = self.gaussian_pdf_2d(pos_iou_mean0, pos_iou_var, self.cov, self.pos_score_n) \
        #           + self.gaussian_pdf_2d(pos_iou_mean1, pos_iou_var, self.cov, self.pos_score_n)  # [C, N_s]

        # a = (pos_iou_mean[:, 0] + pos_iou_mean[:, 1]) * 0.5
        # b = torch.sqrt(pos_iou_var[:, 0] * pos_iou_var[:, 1])
        # pos_iou_mean = torch.stack([a, a], dim=1)
        # pos_iou_var = torch.stack([b, b], dim=1)
        # cov = b * 0.99
        # t1_fake = self.gaussian_pdf_2d(pos_iou_mean, pos_iou_var, cov, self.pos_score_n) * 2

        return t1_fake

    def get_mAP(self, t1, t2, t1_fake):
        t1 = t1 + t1_fake * self.train_cfg.lead_ratio

        dR = torch.cumsum(t1, dim=2)  # [C, N_s, N_i]
        TP = torch.cumsum(dR, dim=1)  # [C, N_s, N_i]
        NP = torch.cumsum(t2, dim=1) + TP[:, :, -1]  # [C, N_s]
        PdR = TP * dR / (NP.unsqueeze(-1) + np.spacing(1))  # [C, N_s, N_i]
        PdR = torch.sum(PdR * self.iou_weight, dim=2, keepdim=False) / self.iou_weight.sum()
        AP = torch.sum(PdR, dim=1, keepdim=False) / (TP[:, -1, -1].detach() + np.spacing(1))
        mAP = torch.sum(AP, dim=0, keepdim=False) * 100 / self.num_classes

        dR = torch.sum(dR, dim=0, keepdim=True)  # [1, N_s, N_i]
        TP = torch.sum(TP, dim=0, keepdim=True)  # [1, N_s, N_i]
        NP = torch.sum(NP, dim=0, keepdim=True)  # [1, N_s]

        PdR = TP * dR / (NP.unsqueeze(-1) + np.spacing(1))  # [1, N_s, N_i]
        PdR = torch.sum(PdR * self.iou_weight, dim=2, keepdim=False) / self.iou_weight.sum()
        AP = torch.sum(PdR, dim=1, keepdim=False) / (TP[:, -1, -1].detach() + np.spacing(1))
        mAP_cat = AP[0] * 100
        if self.train_cfg.class_cat == 'switch':
            mAP = mAP * self.epoch + mAP_cat * (1 - self.epoch)
        elif self.train_cfg.class_cat == 'cat':
            mAP = mAP_cat
        elif self.train_cfg.class_cat == 'none':
            mAP = mAP
        else:
            raise RuntimeError('train_cfg.class_cat error')
        return mAP

    def get_delta_map(self, dR, TP, NP, new_one):
        PdR = (TP * dR / (NP + np.spacing(1))) * self.iou_weight  # [C, N_s, N_i]
        PdR2 = (TP * dR / (NP + new_one + np.spacing(1))) * self.iou_weight  # [C, N_s, N_i]
        PdR3 = (new_one * dR / (NP + new_one + np.spacing(1))) * self.iou_weight  # [C, N_s, N_i]
        PdR4 = (new_one * (TP + new_one) / (NP + new_one + np.spacing(1))) * self.iou_weight  # [C, N_s, N_i]

        PdR = torch.sum(PdR, dim=2, keepdim=True)
        PdR = torch.cumsum(PdR, dim=1)
        ori_mAP = PdR[:, -1:, :]  # [C, 1, 1]
        PdR = torch.cat([PdR.new_zeros([self.num_classes, 1, 1]), PdR[:, :-1, :]], dim=1)  # [C, N_s, 1]

        PdR2 = torch.sum(PdR2, dim=2, keepdim=True)
        PdR2 = torch.cumsum(PdR2.flip(dims=[1, ]), dim=1).flip(dims=[1, ])  # [C, N_s, 1]

        PdR3 = torch.cumsum(torch.cumsum(PdR3.flip(dims=[1, 2]), dim=1), dim=2).flip(dims=[1, 2])  # [C, N_s, N_i]

        PdR4 = torch.cumsum(PdR4.flip(dims=[2, ]), dim=2).flip(dims=[2, ])  # [C, N_s, N_i]

        mAP_t1 = PdR + PdR2 + PdR3 + PdR4
        mAP_t1 = mAP_t1 / self.iou_weight.sum() / (TP[:, -1:, -1:] + new_one + np.spacing(1))
        ori_mAP = ori_mAP / self.iou_weight.sum() / (TP[:, -1:, -1:] + np.spacing(1))
        return mAP_t1 - ori_mAP

    def get_LA_mat(self, t1_fake):
        t1 = t1_fake * self.train_cfg.lead_ratio  # [C, N_s]
        new_one = FACTOR  # * (1 - self.train_cfg.momentum)
        dR = torch.cumsum(t1, dim=2)  # [C, N_s, N_i]
        TP = torch.cumsum(dR, dim=1)  # [C, N_s, N_i]
        NP = TP[:, :, -1:]  # [C, N_s, 1]
        # NP = NP + torch.cumsum(t2, dim=1).unsqueeze(-1)
        mAP = self.get_delta_map(dR, TP, NP, new_one)
        # dR = torch.sum(dR, dim=0, keepdim=True)  # [1, N_s, N_i]
        # TP = torch.sum(TP, dim=0, keepdim=True)  # [1, N_s, N_i]
        # NP = torch.sum(NP, dim=0, keepdim=True)  # [1, N_s, 1]
        # mAP_cat = self.get_delta_map(dR, TP, NP, new_one)
        #
        # if self.train_cfg.class_cat_beta == 0:
        #     mAP = mAP
        # elif self.train_cfg.class_cat_beta == 'inf':
        #     mAP = mAP_cat
        # else:
        #     a = math.pow(self.epoch, self.train_cfg.class_cat_beta)
        #     mAP = mAP * a + mAP_cat * (1 - a)
        # print(mAP)
        mAP_min = mAP.min()
        mAP_max = mAP.max()
        mAP = (mAP - mAP_min) / (mAP_max - mAP_min) * 0.98 + 0.01
        self.mAP_matrix = mAP

    def gaussian_pdf_1d(self, mean, var, N):
        x = self.s_benckmarch.unsqueeze(0)
        PDF = torch.exp(-torch.square((x - mean.unsqueeze(1)) / (torch.sqrt(2 * var.unsqueeze(1))))) / (
            torch.sqrt(2 * math.pi * var.unsqueeze(1)))
        return PDF * N.unsqueeze(1) * self.ds * FACTOR

    def gaussian_pdf_2d(self, pos_iou_mean, pos_iou_var, cov, pos_score_n):

        xy_minus_mu = self.s_iou_benckmarch.flatten(0, 1).unsqueeze(0) - pos_iou_mean.unsqueeze(1)  # [C, N, 2]

        det_C = (pos_iou_var[:, 0] * pos_iou_var[:, 1] - cov.square()).clamp_min(1e-8)  # [C, ]

        constant = 1 / (2 * np.pi * torch.sqrt(det_C))  # [C, ]
        exp = 0.5 * torch.sum(xy_minus_mu.square() * pos_iou_var.flip(dims=[1]).unsqueeze(1), dim=2, keepdim=False) \
              - xy_minus_mu[:, :, 0] * xy_minus_mu[:, :, 1] * cov.unsqueeze(1)  # [C, N]
        mv_normal = constant.unsqueeze(1) * torch.exp(-exp.clamp_min(0) / det_C.unsqueeze(1))  # [C, N]
        t1 = mv_normal.view(pos_iou_mean.shape[0], *self.s_iou_benckmarch.shape[:2]) * self.iou_trans_weight.view(1, 1,
                                                                                                                  -1)  # [C, N_score, N_iou]
        return t1 * pos_score_n.view(-1, 1, 1) * self.ds * self.di * FACTOR

    # @profile
    def gather_dist_Ngt(self, *t):
        if dist.is_initialized():
            for i in t:
                dist.all_reduce(i)
        return t

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        score_factors: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            score_factors (list[Tensor], optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Defaults to None.
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds)
        assert len(cls_scores) == len(bbox_preds)
        cfg = self.test_cfg if cfg is None else cfg
        nms_pre = cfg.get('nms_pre', 1000)
        score_thr = cfg.get('score_thr', 0)
        flatten_cls_scores, flatten_bbox, flatten_points, all_level_points = self.decode_bbox(cls_scores, bbox_preds)
        flatten_cls_scores, flatten_bbox, flatten_label = self.score_topk(flatten_cls_scores, flatten_bbox,
                                                                          all_level_points,
                                                                          nms_pre)
        flatten_bbox = flatten_bbox.transpose(1, 2)
        flatten_cls_scores = flatten_cls_scores.sigmoid()
        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            results = InstanceData()
            results.bboxes = flatten_bbox[img_id]
            results.scores = flatten_cls_scores[img_id]
            results.labels = flatten_label[img_id]
            results = results[results.scores > score_thr]
            if rescale:
                assert img_meta.get('scale_factor') is not None
                scale_factor = [1 / s for s in img_meta['scale_factor']]
                results.bboxes = scale_boxes(results.bboxes, scale_factor)
            # TODO: deal with `with_nms` and `nms_cfg=None` in test_cfg
            if with_nms and results.bboxes.numel() > 0:
                bboxes = get_box_tensor(results.bboxes)
                det_bboxes, keep_idxs = batched_nms(bboxes, results.scores, results.labels, cfg.nms)
                results = results[keep_idxs]
                # some nms would reweight the score, such as softnms
                results.scores = det_bboxes[:, -1]
                results = results[:cfg.max_per_img]
            result_list.append(results)
        return result_list

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_predictor()
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        # self.scales2 = nn.ModuleList([Scale(1.0) for _ in self.strides])
        # self.register_parameter('centerness_weight', nn.Parameter(torch.ones([1, self.num_classes, 1, 1]), True))

    def get_target(self, batch_gt_instances):
        max_num_bbox = max([i.labels.shape[0] for i in batch_gt_instances])
        batch_bbox = batch_gt_instances[0].bboxes.new_zeros([len(batch_gt_instances), max_num_bbox, 4])
        batch_label = batch_gt_instances[0].labels.new_zeros([len(batch_gt_instances), max_num_bbox])
        pad_flag = batch_gt_instances[0].labels.new_zeros([len(batch_gt_instances), max_num_bbox], dtype=torch.bool)
        for b_id, gt_instances in enumerate(batch_gt_instances):
            batch_bbox[b_id, :gt_instances.bboxes.shape[0], :] = gt_instances.bboxes
            batch_label[b_id, :gt_instances.labels.shape[0]] = gt_instances.labels
            pad_flag[b_id, :gt_instances.labels.shape[0]] = True
        return batch_bbox, batch_label, pad_flag  # [b, n_gt, 4], [b, n_gt], [b, n_gt]

    def score_topk(self, flatten_cls_scores, flatten_bbox, all_level_points, k):
        num_imgs, num_class, num_pred = flatten_cls_scores.shape
        # fpn = [num_pred, ]
        fpn = [i.shape[0] for i in all_level_points]
        flatten_cls_scoress = torch.split(flatten_cls_scores, fpn, dim=2)
        flatten_bboxs = torch.split(flatten_bbox, fpn, dim=2)
        flatten_cls_scoresss = []
        flatten_bboxss = []
        labelss = []
        for i, (flatten_cls_scores, flatten_bbox) in enumerate(zip(flatten_cls_scoress, flatten_bboxs)):
            # if i != 0:
            #     continue
            _, _, num_pred = flatten_cls_scores.shape
            flatten_cls_scores = flatten_cls_scores.reshape(num_imgs, -1)
            if k > 0:
                if k > flatten_cls_scores.shape[1]:
                    k = flatten_cls_scores.shape[1]
                flatten_cls_scores, topk_index = torch.topk(flatten_cls_scores, k, dim=1)
                label = topk_index // num_pred
                topk_index = topk_index % num_pred
                topk_index = topk_index.unsqueeze(1).expand([num_imgs, 4, k])
                flatten_bbox = torch.gather(flatten_bbox, 2, topk_index)
            else:
                raise RuntimeError('k must > 0')
            flatten_cls_scoresss.append(flatten_cls_scores)
            flatten_bboxss.append(flatten_bbox)
            labelss.append(label)

        flatten_cls_scores = torch.cat(flatten_cls_scoresss, 1)
        flatten_bbox = torch.cat(flatten_bboxss, 2)
        label = torch.cat(labelss, 1)
        # flatten_cls_scores = flatten_cls_scores.view(num_imgs, -1)
        # flatten_cls_scores, topk_index = torch.topk(flatten_cls_scores, k, dim=1)
        # label = topk_index // num_pred
        # topk_index = topk_index % num_pred
        # topk_index = topk_index.unsqueeze(1).expand([num_imgs, 4, k])
        # flatten_bbox = torch.gather(flatten_bbox, 2, topk_index)
        return flatten_cls_scores, flatten_bbox, label  # [b, k], [b, 4, k], [b, k]

    def compute_iou(self, bbox1, bbox2, one2one=False):
        # bbox1: [...,n1, 4]  bbox2: [...,n2, 4]
        x_min1, y_min1, x_max1, y_max1 = torch.chunk(bbox1, 4, dim=-1)
        x_min2, y_min2, x_max2, y_max2 = torch.chunk(bbox2, 4, dim=-1)
        if one2one:
            inter_w = (torch.minimum(x_max1, x_max2) - torch.maximum(x_min1, x_min2)).clamp_min(0)
            inter_h = (torch.minimum(y_max1, y_max2) - torch.maximum(y_min1, y_min2)).clamp_min(0)
            intersection = inter_w * inter_h
            area1 = (x_max1 - x_min1).clamp_min(0.1) * (y_max1 - y_min1).clamp_min(0.1)
            area2 = (x_max2 - x_min2).clamp_min(0.1) * (y_max2 - y_min2).clamp_min(0.1)
            union = area1 + area2 - intersection
            ious = (intersection / union)  # [...,1]
            ious = ious.squeeze(-1)
        else:
            inter_w = (torch.minimum(x_max1, x_max2.transpose(-2, -1)) - torch.maximum(x_min1,
                                                                                       x_min2.transpose(-2,
                                                                                                        -1))).clamp_min(
                0)
            inter_h = (torch.minimum(y_max1, y_max2.transpose(-2, -1)) - torch.maximum(y_min1,
                                                                                       y_min2.transpose(-2,
                                                                                                        -1))).clamp_min(
                0)
            intersection = inter_w * inter_h  # [...,n1,n2]
            area1 = (x_max1 - x_min1).clamp_min(0.1) * (y_max1 - y_min1).clamp_min(0.1)
            area2 = (x_max2 - x_min2).clamp_min(0.1) * (y_max2 - y_min2).clamp_min(0.1)
            union = area1 + area2.transpose(-2, -1) - intersection
            ious = (intersection / union)  # [...,n1,n2]
        return ious

    def _init_cls_convs(self) -> None:
        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def _init_reg_convs(self) -> None:
        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def _init_predictor(self) -> None:
        """Initialize predictor layers of the head."""
        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)
        # self.conv_cls = ConvModule(
        #     self.feat_channels, self.cls_out_channels, 3, padding=1, norm_cfg=dict(
        #         type='GN', num_groups=1, requires_grad=True, affine=False), act_cfg=None)

        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        if self.with_centerness:
            self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        # self.conv_centerness = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Hack some keys of the model state dict so that can load checkpoints
        of previous version."""
        version = local_metadata.get('version', None)
        if version is None:
            # the key is different in early versions
            # for example, 'fcos_cls' become 'conv_cls' now
            bbox_head_keys = [
                k for k in state_dict.keys() if k.startswith(prefix)
            ]
            ori_predictor_keys = []
            new_predictor_keys = []
            # e.g. 'fcos_cls' or 'fcos_reg'
            for key in bbox_head_keys:
                ori_predictor_keys.append(key)
                key = key.split('.')
                if len(key) < 2:
                    conv_name = None
                elif key[1].endswith('cls'):
                    conv_name = 'conv_cls'
                elif key[1].endswith('reg'):
                    conv_name = 'conv_reg'
                elif key[1].endswith('centerness'):
                    conv_name = 'conv_centerness'
                else:
                    conv_name = None
                if conv_name is not None:
                    key[1] = conv_name
                    new_predictor_keys.append('.'.join(key))
                else:
                    ori_predictor_keys.pop(-1)
            for i in range(len(new_predictor_keys)):
                state_dict[new_predictor_keys[i]] = state_dict.pop(
                    ori_predictor_keys[i])
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually contain classification scores and bbox predictions.

            - cls_scores (list[Tensor]): Box scores for each scale level, \
            each is a 4D-tensor, the channel number is \
            num_points * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for each scale \
            level, each is a 4D-tensor, the channel number is num_points * 4.
        """
        return multi_apply(self.forward_single, x, self.scales
                           )[:2]

    def decode_bbox(self, cls_scores, bbox_preds):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes, dtype=bbox_preds[0].dtype, device=bbox_preds[0].device, with_stride=True)
        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.view(num_imgs, self.cls_out_channels, -1)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.view(num_imgs, 4, -1)
            for bbox_pred in bbox_preds
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=2)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=2)
        flatten_points = torch.cat([points for points in all_level_points], dim=0).transpose(0, 1).unsqueeze(
            0)  # [1, 4, HW]

        # flatten_bbox_preds.fill_(0.0001)
        x1y1 = flatten_points[:, :2, :] - flatten_bbox_preds[:, :2, :] * flatten_points[:, 2:, :]
        x2y2 = flatten_points[:, :2, :] + flatten_bbox_preds[:, 2:, :] * flatten_points[:, 2:, :]
        flatten_bbox = torch.cat([x1y1, x2y2], 1)

        # b, c, n = flatten_cls_scores.shape
        # flatten_cls_scores = flatten_cls_scores.view(b, c * n)
        # mean = flatten_cls_scores.mean(-1, keepdim=True)
        # var = flatten_cls_scores.var(-1, keepdim=True)
        # flatten_cls_scores = (flatten_cls_scores - mean) / (var + 1e-5).sqrt()
        # flatten_cls_scores = flatten_cls_scores.view(b, c, n)
        return flatten_cls_scores, flatten_bbox, flatten_points, all_level_points  # [b, c, n], [b, 4, n], [1, 4, n], listof[n, 4]

    def save_csv(self, tensor, name):
        vis = Visualizer.get_current_instance()
        tb = vis.get_backend('TensorboardVisBackend')
        numpy_data = tensor.detach().cpu().numpy()
        csv_file_path = os.path.join(tb._save_dir, name + '.csv')
        # 使用CSV库将NumPy数组保存到CSV文件中
        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            # 逐行写入数据
            csv_writer.writerow(numpy_data)

    def aug_test(self,
                 aug_batch_feats: List[Tensor],
                 aug_batch_img_metas: List[List[Tensor]],
                 rescale: bool = False) -> List[ndarray]:
        """Test function with test time augmentation.

        Args:
            aug_batch_feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            aug_batch_img_metas (list[list[dict]]): the outer list indicates
                test-time augs (multiscale, flip, etc.) and the inner list
                indicates images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[ndarray]: bbox results of each class
        """
        return self.aug_test_bboxes(
            aug_batch_feats, aug_batch_img_metas, rescale=rescale)


class ClampGradientFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, lower_bound, upper_bound):
        # 保存输入，它将在反向传播时用于计算梯度
        ctx.save_for_backward(input_tensor)
        ctx.lower_bound = lower_bound
        ctx.upper_bound = upper_bound
        # 返回输入本身，因为前向传播没有任何改变
        return input_tensor.clamp(lower_bound, upper_bound)

    @staticmethod
    def backward(ctx, grad_output):
        # 获取保存的输入和上下界
        input_tensor, = ctx.saved_tensors
        lower_bound = ctx.lower_bound
        upper_bound = ctx.upper_bound

        # 截断梯度
        grad_input = grad_output.clone()
        grad_input[(input_tensor < lower_bound) & (grad_output > 0)] = 0  # 正梯度小于下界时截断为0
        grad_input[(input_tensor > upper_bound) & (grad_output < 0)] = 0  # 负梯度大于上界时截断为0

        return grad_input, None, None  # 返回截断后的梯度和None，因为没有关于下界和上界的梯度


class ScoreCenternessFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cls_score, centerness):
        cls_score_exp = cls_score.exp()
        centerness_exp = centerness.exp()
        ctx.save_for_backward(cls_score_exp, centerness_exp)
        res = torch.log(torch.clamp_min((cls_score_exp * centerness_exp) / (cls_score_exp + centerness_exp + 1), 1e-7))
        return res

    @staticmethod
    def backward(ctx, grad_output):
        # 获取保存的输入和上下界
        cls_score_exp, centerness_exp = ctx.saved_tensors
        grad_score = (centerness_exp + 1) / (cls_score_exp + centerness_exp + 1) * grad_output
        grad_centerness = (cls_score_exp + 1) / (cls_score_exp + centerness_exp + 1) * grad_output
        grad_centerness = torch.sum(grad_centerness, dim=1, keepdim=True)
        return grad_score, grad_centerness

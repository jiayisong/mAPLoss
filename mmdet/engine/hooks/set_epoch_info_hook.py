# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook
from mmengine.model.wrappers import is_model_wrapper
from mmdet.registry import HOOKS


@HOOKS.register_module()
class SetEpochInfoHook(Hook):
    """Set runner's epoch information to the model."""

    def before_train_epoch(self, runner):
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        model.bbox_head.set_epoch(runner.epoch, runner.max_epochs)

    # def before_train_iter(self,
    #                       runner,
    #                       batch_idx: int,
    #                       data_batch = None) -> None:
    #     model = runner.model
    #     if is_model_wrapper(model):
    #         model = model.module
    #     model.bbox_head.set_iter(runner.iter, runner.max_iters)

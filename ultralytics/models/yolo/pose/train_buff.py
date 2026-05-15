# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Buff pose training with WingLoss + sigma tightening + IoU-weighted soft labels.

Extends standard PoseTrainer, keeping pretrained-weight loading intact,
while replacing the default OKS loss with armor-style optimizations.
"""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ultralytics.models.yolo.pose.train import PoseTrainer
from ultralytics.utils import DEFAULT_CFG, LOGGER


class BuffPoseTrainer(PoseTrainer):
    """PoseTrainer for buff (大能量机关) with WingLoss corner-accuracy optimizations.

    Key changes over standard PoseTrainer:
    - WingLoss for small-error sensitivity (better corner precision)
    - sigma=0.05 for tighter keypoint constraints
    - IoU-weighted soft labels for classification
    - Staged training: WingLoss → L1 after 40% epochs

    Pretrained weights from yolo11l-pose.pt are loaded normally;
    only the loss criterion is swapped after model creation.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "pose"
        super().__init__(cfg, overrides, _callbacks)

        self._l1_enabled = False
        self.add_callback("on_train_epoch_start", self._staged_training_callback)

    def _setup_train(self):
        """Build model (with pretrained weights), then swap in WingLoss criterion."""
        super()._setup_train()  # self.model is now a PoseModel, not a str
        _model = _unwrap(self.model)
        _model.init_criterion = lambda: _make_buff_loss(_model)

    # ---- staged training -------------------------------------------------
    def _staged_training_callback(self, trainer):
        if self._l1_enabled:
            return
        staged_epoch = int(self.epochs * 0.4)
        if self.epoch >= staged_epoch:
            try:
                c = _unwrap(trainer.model).criterion
                if hasattr(c, "enable_l1_finetuning"):
                    c.enable_l1_finetuning()
                    self._l1_enabled = True
                    LOGGER.info(
                        f"Staged training: WingLoss → L1 at epoch {self.epoch}/{self.epochs}"
                    )
            except Exception:
                pass  # silently skip if criterion hasn't been built yet

    # ---- validation ------------------------------------------------------
    def get_validator(self):
        from ultralytics.models.yolo import pose
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"
        return pose.PoseValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )


# ---- helpers -------------------------------------------------------------

def _unwrap(model):
    """Return the de-paralleled model."""
    from ultralytics.utils.torch_utils import unwrap_model
    return unwrap_model(model)


def _make_buff_loss(model):
    """ArmorPoseLoss with WingLoss, sigma tightening; IoU-weighted cls OFF (stability)."""
    from ultralytics.utils.loss_armor import ArmorPoseLoss
    loss = ArmorPoseLoss(model, use_iou_weighted_cls=False)
    # 9 点 buff 用稍宽松的 sigma=0.07 避免早期震荡
    loss.keypoint_loss.sigmas.fill_(0.07)
    return loss

# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Armor plate pose training module with optimized loss function."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ultralytics.models.yolo.pose.train import PoseTrainer
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import DEFAULT_CFG


class ArmorPoseModel(PoseModel):
    """YOLO pose model optimized for armor plate 4-point detection.

    This model uses an optimized loss function designed specifically for
    armor plate corner keypoint detection, featuring:
    - Smaller sigma values for tighter keypoint constraints
    - Wing Loss for better convergence on small errors
    - Adjusted loss weight ratios

    Example:
        >>> model = ArmorPoseModel("armor-pose-mobilenet.yaml", ch=3, nc=1, data_kpt_shape=(4, 2))
        >>> results = model.predict(image_tensor)
    """

    def init_criterion(self):
        """Initialize the optimized loss criterion for armor plate detection."""
        from ultralytics.utils.loss_armor import ArmorPoseLoss
        return ArmorPoseLoss(self)


class ArmorPoseTrainer(PoseTrainer):
    """Trainer class optimized for armor plate pose estimation.

    This trainer extends PoseTrainer with:
    - Optimized loss function for 4-point keypoint detection
    - Better hyperparameter defaults for armor plate detection
    - Support for smaller input sizes and faster training

    Example:
        >>> from ultralytics.models.yolo.pose.train_armor import ArmorPoseTrainer
        >>> trainer = ArmorPoseTrainer(overrides={
        ...     "model": "armor-pose-mobilenet.yaml",
        ...     "data": "armor_plate.yaml",
        ...     "epochs": 100,
        ...     "imgsz": 640
        ... })
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """Initialize ArmorPoseTrainer with optimized default settings.

        Args:
            cfg (dict): Default configuration dictionary.
            overrides (dict): Dictionary of parameter overrides.
            _callbacks (list): List of callback functions.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "pose"

        # Set optimized defaults for armor plate detection
        armor_defaults = {
            "imgsz": 640,          # Default image size
            "batch": 16,           # Batch size
            "lr0": 0.01,           # Initial learning rate
            "lrf": 0.01,           # Final learning rate factor
            "momentum": 0.937,     # SGD momentum
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "box": 7.5,            # Box loss gain
            "cls": 0.5,            # Cls loss gain
            "dfl": 1.5,            # DFL loss gain
            "pose": 12.0,          # Pose loss gain (increased for armor)
            "kobj": 1.0,           # Keypoint obj loss gain
            "close_mosaic": 10,    # Close mosaic last N epochs
        }

        # Apply defaults only if not already specified
        for key, value in armor_defaults.items():
            if key not in overrides:
                overrides[key] = value

        super().__init__(cfg, overrides, _callbacks)

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> ArmorPoseModel:
        """Get armor pose estimation model with optimized loss function.

        Args:
            cfg (str | Path | dict): Model configuration file path or dictionary.
            weights (str | Path): Path to model weights file.
            verbose (bool): Whether to display model information.

        Returns:
            (ArmorPoseModel): Initialized armor pose estimation model.
        """
        model = ArmorPoseModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            data_kpt_shape=self.data["kpt_shape"],
            verbose=verbose
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return validator instance for model evaluation."""
        from ultralytics.models.yolo import pose
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"
        return pose.PoseValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks
        )

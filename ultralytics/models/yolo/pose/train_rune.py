# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Rune pose training module with rotation-invariant loss function.

This module implements training for rune target detection with 4-fold symmetry:
- RunePoseModel: Pose model using rotation-invariant loss
- RunePoseTrainer: Trainer with optimized defaults for rune detection

The rune has 4-fold central symmetry with 8 keypoints. Due to this symmetry,
the network may confuse the rotational order (0/90/180/270 degrees).
The rotation-invariant loss handles this by computing loss for all 4 rotations
and taking the minimum.
"""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ultralytics.models.yolo.pose.train import PoseTrainer
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import DEFAULT_CFG, LOGGER


class RunePoseModel(PoseModel):
    """YOLO pose model with rotation-invariant loss for rune detection.

    This model uses a specialized loss function that handles the 4-fold
    symmetry of rune targets by computing loss across all 4 cyclic rotations
    and taking the minimum.

    Example:
        >>> model = RunePoseModel("rune-pose-mobilenet.yaml", ch=3, nc=2, data_kpt_shape=(8, 2))
        >>> results = model.predict(image_tensor)
    """

    def init_criterion(self):
        """Initialize the rotation-invariant loss criterion for rune detection."""
        from ultralytics.utils.loss_rune import RunePoseLoss
        return RunePoseLoss(self)


class RunePoseTrainer(PoseTrainer):
    """Trainer class for rune pose estimation with rotation-invariant loss.

    This trainer extends PoseTrainer with:
    - Rotation-invariant loss function for 8-keypoint rune targets
    - Optimized hyperparameters for rune detection
    - High pose loss weight for accurate keypoint localization

    Example:
        >>> from ultralytics.models.yolo.pose.train_rune import RunePoseTrainer
        >>> trainer = RunePoseTrainer(overrides={
        ...     "model": "rune-pose-mobilenet.yaml",
        ...     "data": "rune.yaml",
        ...     "epochs": 100,
        ...     "imgsz": 640
        ... })
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """Initialize RunePoseTrainer with optimized default settings.

        Args:
            cfg (dict): Default configuration dictionary.
            overrides (dict): Dictionary of parameter overrides.
            _callbacks (list): List of callback functions.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "pose"

        # Set optimized defaults for rune detection
        rune_defaults = {
            "imgsz": 640,           # Default image size
            "batch": 16,            # Batch size
            "lr0": 0.01,            # Initial learning rate
            "lrf": 0.01,            # Final learning rate factor
            "momentum": 0.937,      # SGD momentum
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "box": 7.5,             # Box loss gain
            "cls": 0.5,             # Cls loss gain
            "dfl": 1.5,             # DFL loss gain
            "pose": 80.0,           # Pose loss gain (high weight for accurate keypoints)
            "kobj": 1.0,            # Keypoint obj loss gain
            "close_mosaic": 10,     # Close mosaic last N epochs
        }

        # Apply defaults only if not already specified
        for key, value in rune_defaults.items():
            if key not in overrides:
                overrides[key] = value

        super().__init__(cfg, overrides, _callbacks)

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> RunePoseModel:
        """Get rune pose estimation model with rotation-invariant loss.

        Args:
            cfg (str | Path | dict): Model configuration file path or dictionary.
            weights (str | Path): Path to model weights file.
            verbose (bool): Whether to display model information.

        Returns:
            (RunePoseModel): Initialized rune pose estimation model.
        """
        model = RunePoseModel(
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

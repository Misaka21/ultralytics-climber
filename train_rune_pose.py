#!/usr/bin/env python3
"""Rune pose training entrypoint with cyclic-rotation-invariant loss."""

from ultralytics import YOLO, settings
from ultralytics.models.yolo.pose.train_rune import RunePoseTrainer

# 先启用 TensorBoard（必须在构建模型前）
settings.update(tensorboard=True)

# 可直接改这三个 YAML 路径
MODEL_YAML = "config/models/rune/rune-pose-mobilenet.yaml"
DATA_YAML = "config/datasets/rune.yaml"
HYP_YAML = "config/hyperparams/rune_pose.yaml"


class RuneYOLO(YOLO):
    """Use RunePoseTrainer while keeping YOLO(...).train(...) style."""

    @property
    def task_map(self):
        mapping = super().task_map
        mapping["pose"]["trainer"] = RunePoseTrainer
        return mapping


# 加载 rune pose 模型配置（8点 + 旋转不变损失）
model = RuneYOLO(MODEL_YAML)

# 开始训练
model.train(
    data=DATA_YAML,
    cfg=HYP_YAML,   # 读取超参数 YAML
    epochs=300,
    batch=16,
    imgsz=640,
    device=0,
    workers=8,
    project="runs/pose",
    name="rune_rotation_invariant",
    val=True,
    patience=50,
    save=True,
    plots=True,
    # 这些参数会覆盖 HYP_YAML 中同名项
    flipud=0.0,
    fliplr=0.0,  # 避免破坏左右顺序约束
    mosaic=0.5,
    mixup=0.0,
    copy_paste=0.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
)

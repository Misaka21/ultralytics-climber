#!/usr/bin/env python3
"""Armor pose training entrypoint (kpt_shape=[4,2], no rune rotation invariance)."""

from ultralytics import YOLO, settings
from ultralytics.models.yolo.pose.train_armor import ArmorPoseTrainer

# 先启用 TensorBoard（必须在构建模型前）
settings.update(tensorboard=True)

# 可直接改这三个 YAML 路径
MODEL_YAML = "config/models/armor/armor-pose-mobilenet.yaml"
DATA_YAML = "config/datasets/armor_plate.yaml"
HYP_YAML = "config/hyperparams/armor_pose.yaml"


class ArmorYOLO(YOLO):
    """Use ArmorPoseTrainer while keeping YOLO(...).train(...) style."""

    @property
    def task_map(self):
        mapping = super().task_map
        mapping["pose"]["trainer"] = ArmorPoseTrainer
        return mapping


# 加载 armor pose 模型配置（4点角点）
model = ArmorYOLO(MODEL_YAML)

# 开始训练
model.train(
    data=DATA_YAML,
    cfg=HYP_YAML,   # 读取超参数 YAML
    epochs=300,
    batch=32,
    imgsz=640,
    device=0,
    workers=8,
    project="runs/pose",
    name="armor_pose_mobilenet",
    val=True,
    patience=50,
    save=True,
    plots=True,
    # 这些参数会覆盖 HYP_YAML 中同名项
    fliplr=0.5,     # 对应 armor_plate.yaml 里的 flip_idx: [1, 0, 3, 2]
    flipud=0.0,
    mosaic=0.2,
    mixup=0.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.3,
    shear=0.0,
    perspective=0.0,
)

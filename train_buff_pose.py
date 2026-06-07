#!/usr/bin/env python3
"""大能量机关（Buff）Pose 训练脚本 — 高精度模型，无翻转，角点优先."""

import subprocess
from ultralytics import YOLO, settings
from ultralytics.models.yolo.pose.train_buff import BuffPoseTrainer

settings.update(tensorboard=True)

MODEL_YAML = "config/models/buff/buff-pose-yolo11.yaml"   # 用 YAML 构建（含 SpatialAttention in Pose head）
PRETRAINED = "yolo11l-pose.pt"   # COCO 预训练权重，匹配的层自动迁移
DATA_YAML = "config/datasets/buff.yaml"
HYP_YAML = "config/hyperparams/buff_pose.yaml"


class BuffYOLO(YOLO):
    """Hook BuffPoseTrainer into the standard YOLO().train() flow."""

    @property
    def task_map(self):
        mapping = super().task_map
        mapping["pose"]["trainer"] = BuffPoseTrainer
        return mapping


def main():
    model = BuffYOLO(MODEL_YAML)   # 从 YAML 构建，C3k2_CoordAtt + CoordAtt/CoordConv 全部生效
    model.load(PRETRAINED)        # 加载 COCO 预训练权重（新增层随机初始化）

    model.train(
        data=DATA_YAML,
        cfg=HYP_YAML,
        epochs=200,
        batch=16,
        imgsz=640,
        device=0,
        workers=8,
        amp=True,
        project="runs/pose",
        name="buff_pose_yolo11l",
        val=True,
        patience=80,
        save=True,
        plots=True,
        optimizer="AdamW",

        # 微调学习率：预训练模型用更小的 lr
        lr0=0.001,
        lrf=0.01,

        # 损失权重：pose 高权重保证角点回归精度
        cls=0.5,
        pose=80.0,

        # 严格关闭翻转
        flipud=0.0,
        fliplr=0.0,

        # 保守增强
        mosaic=0.2,
        close_mosaic=50,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.01,
        hsv_s=0.15,
        hsv_v=0.25,
        degrees=3.0,
        translate=0.05,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
        erasing=0.0,
    )


if __name__ == "__main__":
    main()

    # 2 分钟后关机
    subprocess.run(["shutdown", "-h", "+2"], check=False)

#!/usr/bin/env python3
"""大能量机关（Buff）Pose 训练脚本 — 高精度模型，无翻转，角点优先."""

import subprocess
from ultralytics import YOLO, settings

settings.update(tensorboard=True)

# 从 COCO 预训练权重开始迁移学习，backbone+neck 已有通用特征
PRETRAINED = "yolo11l-pose.pt"   # 首次运行自动下载；想换 x 改 yolo11x-pose.pt
DATA_YAML = "config/datasets/buff.yaml"
HYP_YAML = "config/hyperparams/buff_pose.yaml"


def main():
    model = YOLO(PRETRAINED)  # 加载 COCO 预训练权重，head 根据 data 自动适配 nc/kpt_shape

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
        pose=120.0,

        # 严格关闭翻转
        flipud=0.0,
        fliplr=0.0,

        # 保守增强
        mosaic=0.2,
        close_mosaic=100,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
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

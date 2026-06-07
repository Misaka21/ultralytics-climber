#!/usr/bin/env python3
"""Buff 轻量训练 — RepVGG nano backbone + WingLoss + sigma 收紧 + 分阶段训练.

与 train_buff_pose.py 共享同一数据集 (buff.yaml)，但不使用 YOLO11L 预训练，
改用 RepVGG 轻量 backbone 从头训练，适合 NUC OpenVINO 部署。
"""

import subprocess
from ultralytics import YOLO, settings
from ultralytics.models.yolo.pose.train_buff import BuffPoseTrainer

settings.update(tensorboard=True)

MODEL_YAML = "config/models/buff/buff-pose-repvgg.yaml"
DATA_YAML = "config/datasets/buff.yaml"
HYP_YAML = "config/hyperparams/buff_pose.yaml"


class BuffYOLO(YOLO):
    """Hook BuffPoseTrainer into YOLO().train() flow."""

    @property
    def task_map(self):
        mapping = super().task_map
        mapping["pose"]["trainer"] = BuffPoseTrainer
        return mapping


def main():
    model = BuffYOLO(MODEL_YAML)   # RepVGG nano，scales 首个 key='n' 即默认

    model.train(
        data=DATA_YAML,
        cfg=HYP_YAML,
        epochs=300,                 # 从头训练需要更多轮次
        batch=128,                  # nano 模型极小，8000 张图 batch=128 无压力
        imgsz=640,
        device=None,                # 自动检测 GPU/CPU
        workers=8,
        amp=True,
        project="runs/pose",
        name="buff_pose_repvgg_n",
        val=True,
        patience=100,
        save=True,
        plots=True,
        optimizer="AdamW",

        # 从头训练，lr 线性缩放：batch 32→128，lr 0.01→0.04
        lr0=0.04,
        lrf=0.01,
        warmup_epochs=5,            # 大 lr 需要更长 warmup

        # 损失权重：pose 高权重确保 9 点回归精度
        cls=0.5,
        pose=80.0,

        # 严格关闭翻转（9 点有序，含 R 标方向）
        flipud=0.0,
        fliplr=0.0,

        # 保守增强（nano 容量小，HSV 稍强防过拟合光照）
        mosaic=0.2,
        close_mosaic=100,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.01,
        hsv_s=0.15,
        hsv_v=0.1,
        degrees=3.0,
        translate=0.05,
        shear=0.0,
        perspective=0.0,
        erasing=0.0,
    )


if __name__ == "__main__":
    main()

    # 2 分钟后关机
    subprocess.run(["shutdown", "-h", "+2"], check=False)

#!/usr/bin/env python3
"""Rune pose training entrypoint with cyclic-rotation-invariant loss."""

import subprocess
from ultralytics import YOLO, settings
from ultralytics.models.yolo.pose.train_rune import RunePoseTrainer

# 必须在构建模型前启用
settings.update(tensorboard=True)

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


def main():
    model = RuneYOLO(MODEL_YAML)

    model.train(
        data=DATA_YAML,
        cfg=HYP_YAML,
        epochs=1000,
        batch=64,
        imgsz=640,
        device=0,
        workers=8,
        amp=False,
        project="runs/pose",
        name="rune_pose_mobilenet",
        val=True,
        patience=50,
        save=True,
        plots=True,
        optimizer="SGD",  # 固定优化器，按下面 lr0/momentum 训练
        # 覆盖 HYP_YAML 同名项（稳健版）
        flipud=0.0,
        fliplr=0.0,  # 避免破坏左右顺序约束
        mosaic=0.1,
        close_mosaic=200,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.015,
        hsv_s=0.1,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.05,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
    )


if __name__ == "__main__":
    main()

    # 2 分钟后关机（Linux）
    subprocess.run(["shutdown", "-h", "+2"], check=False)

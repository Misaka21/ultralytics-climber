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
    model = RuneYOLO("/root/Misaka21/ultralytics-climber/runs/pose/rune_pose_mobilenet_ft/weights/last.pt")

    model.train(
        data=DATA_YAML,
        cfg=HYP_YAML,
        epochs=200,                  # 微调轮数，不必再长跑
        batch=64,
        imgsz=640,
        device=0,
        workers=8,
        amp=False,
        project="runs/pose",
        name="rune_pose_mobilenet_ft_continue",
        val=True,
        patience=0,
        save=True,
        plots=True,
        optimizer="SGD",

        # 建议同时显式固定关键损失权重，避免被 cfg 误覆盖
        cls=0.5,
        pose=80.0,

        # 关键点任务保守增强（提升角点稳定性）
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.3,
        close_mosaic=150,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.05,
        hsv_s=0.12,
        hsv_v=0.25,
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

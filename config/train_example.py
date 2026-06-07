#!/usr/bin/env python3
"""示例：使用配置文件进行训练

Usage:
    python config/train_example.py --task armor_pose
    python config/train_example.py --task armor_detect
"""

import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO


def load_yaml(path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(path) as f:
        return yaml.safe_load(f)


def train(task: str, data_path: str = None, device: str = "0"):
    """根据任务类型训练模型

    Args:
        task: 任务类型，可选 armor_pose, armor_detect
        data_path: 自定义数据集路径（可选）
        device: 训练设备
    """
    config_dir = Path(__file__).parent

    # 任务配置映射
    task_configs = {
        "armor_pose": {
            "model": config_dir / "models/armor/armor-pose-mobilenet.yaml",
            "data": config_dir / "datasets/armor_plate.yaml",
            "hyp": config_dir / "hyperparams/armor_pose.yaml",
        },
        "armor_detect": {
            "model": config_dir / "models/armor/armor-detect-mobilenet.yaml",
            "data": config_dir / "datasets/armor_dataset_v4.yaml",
            "hyp": config_dir / "hyperparams/armor_detect.yaml",
        },
    }

    if task not in task_configs:
        raise ValueError(f"Unknown task: {task}. Available: {list(task_configs.keys())}")

    cfg = task_configs[task]

    # 加载超参数
    hyp = load_yaml(cfg["hyp"])

    # 覆盖自定义数据集路径
    if data_path:
        hyp["data"] = data_path
    else:
        hyp["data"] = str(cfg["data"])

    # 覆盖设备
    hyp["device"] = device

    # 创建模型并训练
    print(f"\n🚀 Training task: {task}")
    print(f"   Model: {cfg['model']}")
    print(f"   Data:  {hyp['data']}")
    print(f"   Hyp:   {cfg['hyp']}\n")

    model = YOLO(str(cfg["model"]))
    model.train(**hyp)

    print(f"\n✅ Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with config files")
    parser.add_argument("--task", type=str, required=True,
                        choices=["armor_pose", "armor_detect"],
                        help="Task to train")
    parser.add_argument("--data", type=str, default=None,
                        help="Override dataset path")
    parser.add_argument("--device", type=str, default="0",
                        help="Device to use (e.g., 0, 0,1, cpu)")

    args = parser.parse_args()

    train(args.task, args.data, args.device)

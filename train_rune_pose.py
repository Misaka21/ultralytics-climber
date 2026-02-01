#!/usr/bin/env python3
"""Train Rune Pose Detection Model with rotation-invariant loss.

Usage:
    python train_rune_pose.py
    python train_rune_pose.py --epochs 200 --batch 32

This script trains a pose estimation model for rune detection with:
- MobileNetV3 lightweight backbone
- 8 keypoints per target with 4-fold rotational symmetry
- Rotation-invariant loss for rune_targeting class
- 4 classes: inactive, targeting, hit, center
"""

from ultralytics.models.yolo.pose.train_rune import RunePoseTrainer


def train_rune_pose(
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    device: str = "0",
):
    """Train rune pose detection model with rotation-invariant loss.

    Args:
        epochs: Number of training epochs.
        batch: Batch size.
        imgsz: Input image size.
        device: Device to use (e.g., "0", "0,1", "cpu").
    """
    # Model and dataset configuration
    model_yaml = "ultralytics/cfg/models/rune/rune-pose-mobilenet.yaml"
    data_yaml = "ultralytics/cfg/datasets/rune.yaml"

    # Create trainer with rotation-invariant loss
    trainer = RunePoseTrainer(overrides={
        "model": model_yaml,
        "data": data_yaml,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,

        # Loss weights (defaults from RunePoseTrainer)
        # box=7.5, cls=0.5, dfl=1.5, pose=80.0, kobj=1.0

        # Augmentation
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 10.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,       # No vertical flip for rune
        "fliplr": 0.5,       # Horizontal flip
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,

        # Training settings
        "close_mosaic": 10,
        "amp": True,

        # Output
        "project": "runs/pose",
        "name": "rune-rotation-invariant",
        "exist_ok": True,

        # Validation
        "val": True,
        "plots": True,
        "save": True,
    })

    # Train the model
    trainer.train()

    print(f"\nTraining complete! Best model saved to: {trainer.save_dir}/weights/best.pt")
    return trainer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Rune Pose Model with Rotation-Invariant Loss")
    parser.add_argument("--device", type=str, default="0", help="Device to use (e.g., 0, 0,1, cpu)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")

    args = parser.parse_args()

    train_rune_pose(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
    )

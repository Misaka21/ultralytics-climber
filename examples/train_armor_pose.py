#!/usr/bin/env python3
"""
Armor Plate Pose Detection Training Script
=========================================

This script demonstrates how to train a lightweight MobileNetV3-based
pose model for armor plate 4-point keypoint detection.

Features:
- MobileNetV3 backbone for fast inference
- Optimized loss function for faster convergence
- 640x640 input resolution
- 4-point keypoint detection (armor plate corners)

Usage:
    python train_armor_pose.py

Requirements:
    - Prepare your dataset in YOLO pose format
    - Update the data path in armor_plate.yaml
"""

from pathlib import Path
from ultralytics.models.yolo.pose.train_armor import ArmorPoseTrainer


def train_armor_pose():
    """Train armor plate pose detection model."""

    # Training configuration
    config = {
        # Model configuration
        "model": "ultralytics/cfg/models/armor/armor-pose-mobilenet.yaml",
        "data": "ultralytics/cfg/datasets/armor_plate.yaml",

        # Training settings
        "epochs": 200,
        "imgsz": 640,
        "batch": 16,
        "workers": 8,
        "device": 0,  # GPU device, use "cpu" for CPU training

        # Optimizer settings (optimized for armor plate detection)
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,

        # Warmup settings
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,

        # Loss weights (optimized for 4-point keypoint detection)
        "box": 7.5,      # Box loss gain
        "cls": 0.5,      # Classification loss gain
        "dfl": 1.5,      # DFL loss gain
        "pose": 12.0,    # Pose/keypoint loss gain (increased for armor)
        "kobj": 1.0,     # Keypoint objectness loss gain

        # Data augmentation
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 10.0,     # Rotation augmentation (limited for armor)
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,       # No vertical flip (armor has orientation)
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,

        # Training behavior
        "close_mosaic": 10,  # Close mosaic in last 10 epochs
        "amp": True,         # Automatic mixed precision
        "patience": 50,      # Early stopping patience
        "save_period": 10,   # Save checkpoint every N epochs

        # Logging
        "project": "runs/armor_pose",
        "name": "mobilenet_640",
        "exist_ok": True,
        "verbose": True,
    }

    # Initialize trainer with optimized settings
    trainer = ArmorPoseTrainer(overrides=config)

    # Start training
    print("=" * 60)
    print("Armor Plate Pose Detection Training")
    print("=" * 60)
    print(f"Model: MobileNetV3 Backbone")
    print(f"Input size: {config['imgsz']}x{config['imgsz']}")
    print(f"Keypoints: 4 (armor plate corners)")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch']}")
    print("=" * 60)

    results = trainer.train()

    print("\nTraining completed!")
    print(f"Best model saved to: {trainer.best}")

    return results


def export_model(weights_path: str, format: str = "onnx"):
    """Export trained model to different formats.

    Args:
        weights_path: Path to trained weights (.pt file)
        format: Export format ('onnx', 'torchscript', 'tflite', etc.)
    """
    from ultralytics import YOLO

    model = YOLO(weights_path)

    # Export with optimization
    model.export(
        format=format,
        imgsz=640,
        half=True,           # FP16 for faster inference
        simplify=True,       # ONNX simplification
        dynamic=False,       # Fixed input size
        opset=12,            # ONNX opset version
    )

    print(f"Model exported to {format} format")


def validate_model(weights_path: str, data_path: str):
    """Validate trained model on test set.

    Args:
        weights_path: Path to trained weights
        data_path: Path to dataset yaml
    """
    from ultralytics import YOLO

    model = YOLO(weights_path)
    results = model.val(data=data_path, imgsz=640)

    print("\nValidation Results:")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    if hasattr(results, 'pose'):
        print(f"Keypoint mAP50: {results.pose.map50:.4f}")
        print(f"Keypoint mAP50-95: {results.pose.map:.4f}")


if __name__ == "__main__":
    # Train the model
    train_armor_pose()

    # Optional: Export and validate
    # export_model("runs/armor_pose/mobilenet_640/weights/best.pt", "onnx")
    # validate_model("runs/armor_pose/mobilenet_640/weights/best.pt", "armor_plate.yaml")

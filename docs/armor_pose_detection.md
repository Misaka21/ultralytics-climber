# 装甲板四点检测 - MobileNetV3 Pose模型

## 概述

本项目为RoboMaster装甲板检测任务提供了一个轻量级的四点关键点检测模型，基于MobileNetV3主干网络，针对装甲板检测场景进行了优化。

## 主要特性

1. **MobileNetV3主干网络** - 轻量高效，适合边缘设备部署
2. **4点关键点检测** - 检测装甲板的四个角点
3. **优化的损失函数** - 使用Wing Loss加快收敛，使用更小的sigma值获得更精确的关键点位置
4. **640x640输入尺寸** - 平衡精度和速度

## 文件结构

```
ultralytics/
├── nn/modules/
│   └── mobilenet.py                    # MobileNetV3模块定义
├── cfg/
│   ├── models/armor/
│   │   └── armor-pose-mobilenet.yaml   # 模型配置文件
│   └── datasets/
│       └── armor_plate.yaml            # 数据集配置模板
├── utils/
│   └── loss_armor.py                   # 优化的损失函数
├── models/yolo/pose/
│   └── train_armor.py                  # 装甲板专用训练器
└── examples/
    └── train_armor_pose.py             # 训练示例脚本
```

## 快速开始

### 1. 准备数据集

数据集格式遵循YOLO Pose格式：

```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── ...
│   └── val/
│       └── ...
└── labels/
    ├── train/
    │   ├── image1.txt
    │   └── ...
    └── val/
        └── ...
```

标签格式（每行一个目标）：

```
class_id x_center y_center width height kpt1_x kpt1_y kpt2_x kpt2_y kpt3_x kpt3_y kpt4_x kpt4_y
```

- 所有坐标都是归一化的（0-1）
- 关键点顺序：左上、右上、右下、左下（顺时针）

### 2. 修改数据集配置

编辑 `ultralytics/cfg/datasets/armor_plate.yaml`，修改数据集路径：

```yaml
path: /your/dataset/path
train: images/train
val: images/val
```

### 3. 训练模型

方式一：使用训练脚本

```bash
python examples/train_armor_pose.py
```

方式二：使用命令行

```bash
yolo pose train model=ultralytics/cfg/models/armor/armor-pose-mobilenet.yaml \
  data=ultralytics/cfg/datasets/armor_plate.yaml \
  epochs=200 imgsz=640 batch=16
```

方式三：使用Python API

```python
from ultralytics.models.yolo.pose.train_armor import ArmorPoseTrainer

trainer = ArmorPoseTrainer(
    overrides={
        "model": "ultralytics/cfg/models/armor/armor-pose-mobilenet.yaml",
        "data": "ultralytics/cfg/datasets/armor_plate.yaml",
        "epochs": 200,
        "imgsz": 640,
        "batch": 16,
    }
)
trainer.train()
```

### 4. 推理

```python
from ultralytics import YOLO

model = YOLO("runs/armor_pose/mobilenet_640/weights/best.pt")
results = model.predict("test_image.jpg", imgsz=640)

for result in results:
    # 获取边界框
    boxes = result.boxes.xyxy
    # 获取关键点 (4个角点)
    keypoints = result.keypoints.xy  # shape: (N, 4, 2)
```

### 5. 导出模型

```python
from ultralytics import YOLO

model = YOLO("runs/armor_pose/mobilenet_640/weights/best.pt")
model.export(format="onnx", imgsz=640, half=True, simplify=True)
```

## 损失函数优化

针对装甲板检测任务，我们对损失函数进行了以下优化：

1. **更小的Sigma值 (0.05)** - 装甲板角点位置相对固定，使用更严格的约束
2. **Wing Loss** - 对小误差有更强的梯度，加快收敛
3. **调整的损失权重** - `pose=12.0`（增加关键点损失权重）

## 模型缩放

模型支持多种尺寸：

- `n` (nano): 最小最快
- `s` (small): 平衡速度和精度
- `m` (medium): 更高精度

在配置文件中通过 `scale` 参数选择：

```bash
yolo pose train model=armor-pose-mobilenet.yaml scale=s ...
```

## 训练技巧

1. **数据增强**：适当的旋转（±10°）、缩放、翻转可以提高泛化能力
2. **学习率**：建议初始学习率0.01，使用余弦退火
3. **批次大小**：根据显存调整，建议16或32
4. **早停**：设置patience=50防止过拟合
5. **混合精度**：开启AMP加速训练

## 常见问题

**Q: 模型收敛慢怎么办？**
A: 检查数据标注质量，增加pose损失权重，使用预训练权重

**Q: 关键点精度不够怎么办？**
A: 降低sigma值，增加pose损失权重，使用更大的输入尺寸

**Q: 如何在嵌入式设备部署？**
A: 导出为ONNX格式，使用TensorRT或OpenVINO加速

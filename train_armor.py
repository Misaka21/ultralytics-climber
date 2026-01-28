# 先启用TensorBoard (必须在导入YOLO之前)
from ultralytics import settings
settings.update(tensorboard=True)

from ultralytics import YOLO

# 加载MobileNet检测模型配置
model = YOLO('ultralytics/cfg/models/armor/armor-detect-mobilenet.yaml')

# 开始训练
model.train(
    data='ultralytics/cfg/datasets/armor_dataset_v4.yaml',
    epochs=500,
    batch=32,  # RTX 2060显存6GB，32会OOM
    device=0,  # 使用GPU 0，多卡可用 device=[0,1]
    workers=8,
    project='runs/detect',
    name='armor_mobilenet',
    val=True,
    patience=20,  # 早停patience
    save=True,
    plots=True,
    # 数据增强
    fliplr=0.0,  # 禁用水平翻转
    flipud=0.0,  # 禁用垂直翻转
    mosaic=0.5,      # mosaic增强概率
    mixup=0.0,       # mixup增强概率
    hsv_h=0.015,     # 色调变化
    hsv_s=0.7,       # 饱和度变化
    hsv_v=0.4,       # 亮度变化
    degrees=30,     # 旋转角度
    translate=0.5,   # 平移
    scale=0.5,       # 缩放


)

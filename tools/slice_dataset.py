#!/usr/bin/env python3
"""
Image Slicing Tool for Large Image Datasets
将大尺寸图像切片为小块，用于训练小目标检测模型

Usage:
    # 命令行模式
    python slice_dataset.py --images /path/to/images --labels /path/to/labels --output /path/to/output

    # 交互模式
    python slice_dataset.py

Author: Claude Code
"""

import argparse
import cv2
import os
import sys
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description='将大尺寸图像切片为小块，保持目标原始像素尺寸',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 基本用法
  python slice_dataset.py --images ./images --labels ./labels --output ./sliced

  # 自定义切片大小和重叠
  python slice_dataset.py --images ./images --labels ./labels --output ./sliced \\
      --tile-size 1280 --overlap 256

  # 处理多个文件夹
  python slice_dataset.py --images ./img1 ./img2 --labels ./lab1 ./lab2 --output ./sliced
        '''
    )

    parser.add_argument('--images', '-i', nargs='+', type=str,
                        help='图像文件夹路径 (可指定多个)')
    parser.add_argument('--labels', '-l', nargs='+', type=str,
                        help='标签文件夹路径 (可指定多个，与images一一对应)')
    parser.add_argument('--output', '-o', type=str,
                        help='输出文件夹路径')
    parser.add_argument('--tile-size', '-t', type=int, default=640,
                        help='切片大小 (默认: 640)')
    parser.add_argument('--overlap', '-p', type=int, default=128,
                        help='切片重叠像素 (默认: 128)')
    parser.add_argument('--min-bbox-ratio', '-m', type=float, default=0.3,
                        help='bbox最小可见比例，低于此值丢弃 (默认: 0.3)')
    parser.add_argument('--keep-empty', '-k', action='store_true',
                        help='保留没有目标的切片 (默认: 不保留)')
    parser.add_argument('--workers', '-w', type=int, default=8,
                        help='并行处理线程数 (默认: 8)')
    parser.add_argument('--ext', '-e', type=str, default='jpg',
                        help='图像文件扩展名 (默认: jpg)')
    parser.add_argument('--interactive', action='store_true',
                        help='强制进入交互模式')

    return parser.parse_args()


def interactive_mode():
    """交互模式：通过用户输入获取参数"""
    print("\n" + "=" * 60)
    print("  图像切片工具 - 交互模式")
    print("=" * 60)

    config = {}

    # 图像文件夹
    print("\n[1/7] 图像文件夹路径")
    print("  提示: 可输入多个路径，用逗号分隔")
    print("  示例: /path/to/images1, /path/to/images2")
    while True:
        images_input = input("  >>> ").strip()
        if images_input:
            config['images'] = [p.strip() for p in images_input.split(',')]
            # 验证路径
            valid = True
            for p in config['images']:
                if not os.path.isdir(p):
                    print(f"  [错误] 路径不存在: {p}")
                    valid = False
            if valid:
                break
        else:
            print("  [错误] 请输入至少一个路径")

    # 标签文件夹
    print("\n[2/7] 标签文件夹路径")
    print("  提示: 与图像文件夹一一对应，用逗号分隔")
    print("  提示: 如果标签和图像在同一文件夹，输入 'same'")
    while True:
        labels_input = input("  >>> ").strip()
        if labels_input.lower() == 'same':
            config['labels'] = config['images']
            break
        elif labels_input:
            config['labels'] = [p.strip() for p in labels_input.split(',')]
            if len(config['labels']) != len(config['images']):
                print(f"  [错误] 标签文件夹数量({len(config['labels'])})与图像文件夹数量({len(config['images'])})不匹配")
                continue
            valid = True
            for p in config['labels']:
                if not os.path.isdir(p):
                    print(f"  [错误] 路径不存在: {p}")
                    valid = False
            if valid:
                break
        else:
            print("  [错误] 请输入路径或 'same'")

    # 输出文件夹
    print("\n[3/7] 输出文件夹路径")
    print("  提示: 会自动创建 images/ 和 labels/ 子文件夹")
    while True:
        output_input = input("  >>> ").strip()
        if output_input:
            config['output'] = output_input
            break
        else:
            print("  [错误] 请输入输出路径")

    # 切片大小
    print("\n[4/7] 切片大小 (像素)")
    print("  提示: 推荐 640 或 1280，直接回车使用默认值 640")
    tile_input = input("  >>> ").strip()
    config['tile_size'] = int(tile_input) if tile_input else 640

    # 重叠大小
    print("\n[5/7] 切片重叠 (像素)")
    print("  提示: 推荐为切片大小的 10-30%，直接回车使用默认值 128")
    overlap_input = input("  >>> ").strip()
    config['overlap'] = int(overlap_input) if overlap_input else 128

    # 最小bbox比例
    print("\n[6/7] 最小bbox可见比例")
    print("  提示: 被切边的bbox保留多少比例，推荐 0.3，直接回车使用默认值")
    min_ratio_input = input("  >>> ").strip()
    config['min_bbox_ratio'] = float(min_ratio_input) if min_ratio_input else 0.3

    # 是否保留空切片
    print("\n[7/7] 是否保留无目标的切片？")
    print("  提示: 输入 y/yes 保留，直接回车或 n/no 不保留")
    keep_input = input("  >>> ").strip().lower()
    config['keep_empty'] = keep_input in ['y', 'yes']

    # 确认配置
    print("\n" + "-" * 60)
    print("  配置确认:")
    print("-" * 60)
    print(f"  图像文件夹: {config['images']}")
    print(f"  标签文件夹: {config['labels']}")
    print(f"  输出文件夹: {config['output']}")
    print(f"  切片大小: {config['tile_size']}x{config['tile_size']}")
    print(f"  重叠大小: {config['overlap']}")
    print(f"  最小bbox比例: {config['min_bbox_ratio']}")
    print(f"  保留空切片: {'是' if config['keep_empty'] else '否'}")
    print("-" * 60)

    confirm = input("\n  确认开始处理? (y/n) >>> ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("  已取消")
        sys.exit(0)

    return config


def slice_single_image(args):
    """处理单张图像的切片"""
    img_path, label_path, output_images_dir, output_labels_dir, \
        tile_size, overlap, min_bbox_ratio, keep_empty, img_idx = args

    # 读取图像
    img = cv2.imread(str(img_path))
    if img is None:
        return 0, 0, f"无法读取图像: {img_path}"

    h, w = img.shape[:2]

    # 读取标签 (YOLO格式: class x_center y_center width height)
    labels = []
    if label_path and os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x, y, bw, bh = map(float, parts[1:5])
                    # 转换为像素坐标 (x1, y1, x2, y2)
                    x1 = (x - bw/2) * w
                    y1 = (y - bh/2) * h
                    x2 = (x + bw/2) * w
                    y2 = (y + bh/2) * h
                    labels.append([cls, x1, y1, x2, y2])

    stride = tile_size - overlap
    tile_count = 0
    label_count = 0
    base_name = Path(img_path).stem

    # 生成所有切片位置
    y_positions = list(range(0, max(1, h - tile_size + 1), stride))
    x_positions = list(range(0, max(1, w - tile_size + 1), stride))

    # 确保覆盖到右下角
    if y_positions[-1] + tile_size < h:
        y_positions.append(h - tile_size)
    if x_positions[-1] + tile_size < w:
        x_positions.append(w - tile_size)

    for y0 in y_positions:
        for x0 in x_positions:
            x1_tile, y1_tile = x0, y0
            x2_tile, y2_tile = x0 + tile_size, y0 + tile_size

            # 筛选在此tile内的标签
            tile_labels = []
            for cls, x1, y1, x2, y2 in labels:
                # 计算交集
                ix1 = max(x1, x1_tile)
                iy1 = max(y1, y1_tile)
                ix2 = min(x2, x2_tile)
                iy2 = min(y2, y2_tile)

                if ix1 < ix2 and iy1 < iy2:
                    # 计算可见比例
                    orig_area = (x2 - x1) * (y2 - y1)
                    visible_area = (ix2 - ix1) * (iy2 - iy1)
                    visible_ratio = visible_area / orig_area if orig_area > 0 else 0

                    if visible_ratio >= min_bbox_ratio:
                        # 使用交集区域作为新bbox
                        new_cx = ((ix1 + ix2) / 2 - x1_tile) / tile_size
                        new_cy = ((iy1 + iy2) / 2 - y1_tile) / tile_size
                        new_bw = (ix2 - ix1) / tile_size
                        new_bh = (iy2 - iy1) / tile_size

                        # 确保在有效范围内
                        new_cx = max(0, min(1, new_cx))
                        new_cy = max(0, min(1, new_cy))
                        new_bw = max(0.001, min(1, new_bw))
                        new_bh = max(0.001, min(1, new_bh))

                        tile_labels.append(f"{cls} {new_cx:.6f} {new_cy:.6f} {new_bw:.6f} {new_bh:.6f}")

            # 保存切片
            if tile_labels or keep_empty:
                tile = img[y0:y0+tile_size, x0:x0+tile_size]
                tile_name = f"{base_name}_{x0}_{y0}"

                cv2.imwrite(str(output_images_dir / f"{tile_name}.jpg"), tile)

                with open(output_labels_dir / f"{tile_name}.txt", 'w') as f:
                    f.write('\n'.join(tile_labels))

                tile_count += 1
                label_count += len(tile_labels)

    return tile_count, label_count, None


def process_dataset(config):
    """处理整个数据集"""
    images_dirs = config['images']
    labels_dirs = config['labels']
    output_dir = Path(config['output'])
    tile_size = config['tile_size']
    overlap = config['overlap']
    min_bbox_ratio = config['min_bbox_ratio']
    keep_empty = config['keep_empty']
    workers = config.get('workers', 8)
    ext = config.get('ext', 'jpg')

    # 创建输出目录
    output_images_dir = output_dir / 'images'
    output_labels_dir = output_dir / 'labels'
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有图像路径
    all_tasks = []
    for img_dir, label_dir in zip(images_dirs, labels_dirs):
        img_dir = Path(img_dir)
        label_dir = Path(label_dir)

        for img_path in img_dir.glob(f'*.{ext}'):
            label_path = label_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                # 尝试在同一目录查找
                label_path = img_dir / f"{img_path.stem}.txt"

            all_tasks.append((
                img_path, label_path if label_path.exists() else None,
                output_images_dir, output_labels_dir,
                tile_size, overlap, min_bbox_ratio, keep_empty,
                len(all_tasks)
            ))

    print(f"\n找到 {len(all_tasks)} 张图像")
    print(f"输出目录: {output_dir}")
    print(f"切片参数: {tile_size}x{tile_size}, 重叠 {overlap}px")
    print("-" * 60)

    total_tiles = 0
    total_labels = 0
    errors = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(slice_single_image, task): task for task in all_tasks}

        with tqdm(total=len(all_tasks), desc="切片处理中", unit="img") as pbar:
            for future in as_completed(futures):
                tiles, labels, error = future.result()
                total_tiles += tiles
                total_labels += labels
                if error:
                    errors.append(error)
                pbar.update(1)

    # 保存处理信息
    info = {
        'source_dirs': [str(p) for p in images_dirs],
        'label_dirs': [str(p) for p in labels_dirs],
        'output_dir': str(output_dir),
        'tile_size': tile_size,
        'overlap': overlap,
        'min_bbox_ratio': min_bbox_ratio,
        'keep_empty': keep_empty,
        'total_images': len(all_tasks),
        'total_tiles': total_tiles,
        'total_labels': total_labels,
        'errors': errors
    }

    with open(output_dir / 'slice_info.json', 'w') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("  处理完成!")
    print("=" * 60)
    print(f"  原始图像数: {len(all_tasks)}")
    print(f"  生成切片数: {total_tiles}")
    print(f"  标签总数: {total_labels}")
    if errors:
        print(f"  错误数: {len(errors)}")
    print(f"  输出目录: {output_dir}")
    print("=" * 60)

    return info


def main():
    args = parse_args()

    # 判断是否进入交互模式
    if args.interactive or not (args.images and args.labels and args.output):
        config = interactive_mode()
        config['workers'] = 8
        config['ext'] = 'jpg'
    else:
        config = {
            'images': args.images,
            'labels': args.labels,
            'output': args.output,
            'tile_size': args.tile_size,
            'overlap': args.overlap,
            'min_bbox_ratio': args.min_bbox_ratio,
            'keep_empty': args.keep_empty,
            'workers': args.workers,
            'ext': args.ext
        }

    process_dataset(config)


if __name__ == '__main__':
    main()

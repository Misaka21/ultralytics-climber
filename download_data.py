#!/usr/bin/env python3
"""从 Hugging Face 下载数据集到 data/ 文件夹

Usage:
    python download_data.py --dataset username/armor-dataset --output data/armor
    python download_data.py --dataset username/rune-dataset --output data/rune --mirror
    python download_data.py --list  # 查看预设数据集列表
"""

import argparse
import os
import sys
from pathlib import Path

def setup_hf_cache():
    """设置 HF 缓存目录到数据盘"""
    # 优先使用 /data, /mnt/data 等数据盘
    data_roots = ["/data", "/mnt/data", "/home/data", str(Path.home() / "data")]
    
    for root in data_roots:
        if Path(root).exists():
            cache_dir = Path(root) / "huggingface"
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(cache_dir)
            print(f"📁 HF 缓存目录: {cache_dir}")
            return cache_dir
    
    # 默认使用当前目录
    print("⚠️ 未找到数据盘，使用当前目录 ./data/hf_cache")
    return Path("./data/hf_cache")

def download_dataset(repo_id: str, output_dir: str, use_mirror: bool = False, token: str = None):
    """下载 HF 数据集
    
    Args:
        repo_id: HF 数据集 ID，如 "username/dataset-name"
        output_dir: 本地输出目录
        use_mirror: 是否使用国内镜像
        token: HF access token（私有数据集需要）
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("❌ 请先安装: pip install huggingface-hub")
        sys.exit(1)
    
    # 国内镜像
    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("🌐 使用国内镜像: https://hf-mirror.com")
    
    # 设置缓存
    cache_dir = setup_hf_cache()
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 下载数据集: {repo_id}")
    print(f"📂 保存到: {output_path.absolute()}\n")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(output_path),
            resume_download=True,
            local_dir_use_symlinks=False,  # 直接复制，方便管理
            token=token,
        )
        print(f"\n✅ 下载完成！")
        print(f"   数据路径: {output_path.absolute()}")
        print(f"   请修改 config/datasets/*.yaml 中的 path 为此路径")
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n常见问题:")
        print("   1. 网络问题: 尝试添加 --mirror 使用国内镜像")
        print("   2. 权限问题: 私有数据集需要 --token YOUR_TOKEN")
        print("   3. 磁盘空间: 确保 /data 或当前目录有足够空间")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="下载 Hugging Face 数据集")
    parser.add_argument("--dataset", "-d", type=str, help="HF 数据集 ID，如 username/dataset-name")
    parser.add_argument("--output", "-o", type=str, default="data/dataset", help="输出目录（默认: data/dataset）")
    parser.add_argument("--mirror", "-m", action="store_true", help="使用国内镜像 hf-mirror.com")
    parser.add_argument("--token", "-t", type=str, help="HF access token（私有数据集需要）")
    parser.add_argument("--list", "-l", action="store_true", help="显示推荐的预设数据集")
    
    args = parser.parse_args()
    
    # 预设数据集列表
    PRESETS = {
        "armor": {
            "repo": "your-username/robomaster-armor",  # 替换为实际的
            "output": "data/armor",
            "desc": "RoboMaster 装甲板检测数据集"
        },
        "rune": {
            "repo": "your-username/robomaster-rune",
            "output": "data/rune",
            "desc": "RoboMaster 大符数据集"
        },
        "car": {
            "repo": "your-username/robomaster-car",
            "output": "data/car",
            "desc": "RoboMaster 车辆检测数据集"
        }
    }
    
    if args.list:
        print("📋 预设数据集列表（需要在代码中修改 HF ID）：")
        for key, info in PRESETS.items():
            print(f"\n  {key}:")
            print(f"    描述: {info['desc']}")
            print(f"    用法: python download_data.py --dataset {info['repo']} --output {info['output']}")
        return
    
    if not args.dataset:
        print("❌ 请指定数据集 ID，例如：")
        print(f"   python download_data.py --dataset username/dataset-name --output data/armor")
        print(f"\n或使用 --list 查看预设列表")
        sys.exit(1)
    
    download_dataset(args.dataset, args.output, args.mirror, args.token)

if __name__ == "__main__":
    main()

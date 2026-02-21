"""
数据集管理工具

提供统一的实验数据接口，支持：
- TinyShakespeare（经典字符级数据，1MB）
- TinyStories（现代英文，支持取子集）
- 合成数据（用于可视化实验）

注意：
    - 此文件原名为 datasets.py，已重命名为 data_sources.py
    - 重命名原因：避免与 HuggingFace datasets 库产生命名冲突
    - 详见：https://github.com/joyehuang/minimind-notes/pull/20

系统要求：
    - Python 3.10+（使用了类型联合语法 str | list）
    - 依赖：requests（TinyShakespeare）、datasets（TinyStories）

使用示例：
    from modules.common.data_sources import get_experiment_data

    # 获取 TinyShakespeare
    text = get_experiment_data('shakespeare')

    # 获取 TinyStories 子集（10MB）
    texts = get_experiment_data('tinystories', size_mb=10)
"""

import os
import requests
from pathlib import Path
from typing import List, Optional
import json

# 数据缓存目录
DATA_DIR = Path(__file__).parent / 'data'
DATA_DIR.mkdir(exist_ok=True)


def get_experiment_data(
    dataset: str = 'shakespeare',
    size_mb: Optional[float] = None,
    cache: bool = True
) -> str | List[str]:
    """
    获取实验数据

    Args:
        dataset: 数据集名称
            - 'shakespeare': TinyShakespeare (1MB)
            - 'tinystories': TinyStories (可指定大小)
            - 'synthetic': 合成随机数据
        size_mb: 数据大小限制（仅对 tinystories 有效）
        cache: 是否使用缓存

    Returns:
        str: 文本数据（shakespeare, synthetic）
        List[str]: 文本列表（tinystories）
    """

    if dataset == 'shakespeare':
        return _get_shakespeare(cache)
    elif dataset == 'tinystories':
        return _get_tinystories(size_mb or 10, cache)
    elif dataset == 'synthetic':
        return _generate_synthetic(size_mb or 1)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def _get_shakespeare(cache: bool = True) -> str:
    """下载 TinyShakespeare 数据集"""

    cache_file = DATA_DIR / 'tinyshakespeare.txt'

    # 检查缓存
    if cache and cache_file.exists():
        print(f"✅ 从缓存加载 TinyShakespeare: {cache_file}")
        return cache_file.read_text(encoding='utf-8')

    # 下载
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    print(f"📥 下载 TinyShakespeare from {url}")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text = response.text

        # 保存缓存
        if cache:
            cache_file.write_text(text, encoding='utf-8')
            print(f"✅ 已缓存到: {cache_file}")

        print(f"✅ TinyShakespeare 加载完成: {len(text):,} 字符")
        return text

    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("💡 提示: 检查网络连接或手动下载到:", cache_file)
        raise


def _get_tinystories(size_mb: float, cache: bool = True) -> List[str]:
    """
    获取 TinyStories 子集

    注意：需要安装 datasets 库
        pip install datasets
    """

    cache_file = DATA_DIR / f'tinystories_{size_mb}mb.json'

    # 检查缓存
    if cache and cache_file.exists():
        print(f"✅ 从缓存加载 TinyStories: {cache_file}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    # 下载
    try:
        from datasets import load_dataset

        print(f"📥 下载 TinyStories (目标大小: {size_mb} MB)")

        # 加载数据集
        dataset = load_dataset('roneneldan/TinyStories', split='train', streaming=True)

        # 逐步加载直到达到目标大小
        texts = []
        current_size = 0
        target_size = size_mb * 1024 * 1024  # 转换为字节

        for example in dataset:
            text = example['text']
            texts.append(text)
            current_size += len(text.encode('utf-8'))

            if current_size >= target_size:
                break

        # 保存缓存
        if cache:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(texts, f, ensure_ascii=False)
            print(f"✅ 已缓存到: {cache_file}")

        print(f"✅ TinyStories 加载完成: {len(texts):,} 个故事, {current_size / 1024 / 1024:.2f} MB")
        return texts

    except ImportError:
        print("❌ 缺少 datasets 库，请安装: pip install datasets")
        raise
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        raise


def _generate_synthetic(size_mb: float) -> str:
    """
    生成合成数据（用于快速测试）

    生成简单的重复模式，用于验证模型是否能学习
    """

    # 简单模式：重复字母序列
    pattern = "abcdefghijklmnopqrstuvwxyz " * 100

    # 计算需要重复多少次
    target_size = size_mb * 1024 * 1024
    repeats = int(target_size / len(pattern.encode('utf-8'))) + 1

    text = pattern * repeats
    text = text[:int(target_size)]  # 精确截断

    print(f"✅ 生成合成数据: {len(text):,} 字符, {len(text) / 1024 / 1024:.2f} MB")
    return text


def download_all_datasets():
    """一键下载所有数据集（用于初始化）"""

    print("=" * 60)
    print("开始下载所有实验数据集")
    print("=" * 60)

    total_size = 0

    # 1. TinyShakespeare
    print("\n1️⃣ TinyShakespeare")
    text = get_experiment_data('shakespeare')
    size = len(text.encode('utf-8'))
    total_size += size
    print(f"   大小: {size / 1024 / 1024:.2f} MB")

    # 2. TinyStories (10MB 子集)
    print("\n2️⃣ TinyStories (10MB subset)")
    try:
        texts = get_experiment_data('tinystories', size_mb=10)
        size = sum(len(t.encode('utf-8')) for t in texts)
        total_size += size
        print(f"   大小: {size / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"   ⚠️ 跳过 TinyStories: {e}")

    # 3. TinyStories (50MB 子集，用于完整训练)
    print("\n3️⃣ TinyStories (50MB subset)")
    try:
        texts = get_experiment_data('tinystories', size_mb=50)
        size = sum(len(t.encode('utf-8')) for t in texts)
        total_size += size
        print(f"   大小: {size / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"   ⚠️ 跳过 TinyStories 50MB: {e}")

    print("\n" + "=" * 60)
    print(f"✅ 下载完成！总大小: {total_size / 1024 / 1024:.2f} MB")
    print(f"📂 数据位置: {DATA_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    # 测试
    import argparse

    parser = argparse.ArgumentParser(description='数据集管理工具')
    parser.add_argument('--download-all', action='store_true',
                       help='下载所有数据集')
    parser.add_argument('--dataset', type=str, default='shakespeare',
                       choices=['shakespeare', 'tinystories', 'synthetic'],
                       help='测试单个数据集')

    args = parser.parse_args()

    if args.download_all:
        download_all_datasets()
    else:
        data = get_experiment_data(args.dataset)
        if isinstance(data, str):
            print(f"加载的数据: {len(data):,} 字符")
            print(f"前 100 字符: {data[:100]}")
        else:
            print(f"加载的数据: {len(data):,} 个文本")
            print(f"第一个文本: {data[0][:100]}")

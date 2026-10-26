"""测试集缓存生成入口: 读取 test_images/ 生成 datasets/test_dataset.pt。

用法:
    python generate_test_dataset.py
"""

import argparse

import torch

from defect_detection.config import DATASET_DIR, TEST_DATASET_FILE, TEST_IMAGE_DIR
from defect_detection.data.dataset import load_test_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="生成测试集缓存")
    parser.add_argument("--image-dir", default=str(TEST_IMAGE_DIR), help="测试图片目录")
    parser.add_argument("--output", default=str(TEST_DATASET_FILE), help="输出 .pt 文件路径")
    return parser.parse_args()


def main():
    args = parse_args()

    test_dataset = load_test_dataset(args.image_dir)
    if test_dataset is None:
        raise FileNotFoundError(
            f"未能从 {args.image_dir} 加载到测试图片, 请先准备测试样本。")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(test_dataset, args.output)
    print(f"Test dataset has been saved to {args.output}. Shape: {tuple(test_dataset.shape)}")


if __name__ == "__main__":
    main()

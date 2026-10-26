"""VAE-GAN 训练入口。

数据准备:
    1. train_images/ 放训练用正常样本 (可用 cut_targets.py 从原图切割得到);
    2. test_images/ 放留出样本, 先运行 generate_test_dataset.py 生成 datasets/test_dataset.pt
       (缺失时会自动从 train_images/ 随机切出 1% 作为测试集)。

用法:
    python train.py                   # 默认: train_images/, 60 轮, batch 10
    python train.py --epochs 100 --batch-size 8
"""

import argparse

import torch

from defect_detection.config import (
    DISCRIMINATOR_CHECKPOINT,
    TEST_DATASET_FILE,
    TRAIN_IMAGE_DIR,
    VAE_CHECKPOINT,
)
from defect_detection.data.dataset import augment_tensor_dataset, load_image_datasets, load_test_dataset
from defect_detection.training.vae_gan_trainer import VAEGANModelLoader


def parse_args():
    parser = argparse.ArgumentParser(description="VAE-GAN 训练")
    parser.add_argument("--train-dir", default=str(TRAIN_IMAGE_DIR), help="训练图片目录")
    parser.add_argument("--epochs", type=int, default=60, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=10, help="批大小")
    parser.add_argument("--model", default=str(VAE_CHECKPOINT), help="VAE 权重保存路径")
    parser.add_argument("--discriminator", default=str(DISCRIMINATOR_CHECKPOINT),
                        help="判别器权重保存路径")
    parser.add_argument("--test-interval", type=int, default=5, help="重建可视化间隔 (轮)")
    return parser.parse_args()


def main():
    args = parse_args()

    # 训练集: 读取 -> 数据增强 (旋转/裁剪/亮度扰动) -> 统一 512x512 灰度
    dataset = load_image_datasets(args.train_dir)
    augmented_dataset = augment_tensor_dataset(dataset)

    # 测试集: 优先用生成的缓存, 否则现场读取 test_images/
    import os
    if os.path.isfile(TEST_DATASET_FILE):
        test_dataset = torch.load(TEST_DATASET_FILE, weights_only=True)
        print(f"Test dataset has been loaded from {TEST_DATASET_FILE}.")
    else:
        test_dataset = load_test_dataset()
        if test_dataset is None:
            print("No test dataset found; a small split of the train set will be used.")

    model = VAEGANModelLoader(augmented_dataset, test_dataset, args.batch_size,
                              args.model, args.discriminator)
    model.train(args.epochs, args.test_interval)
    model.test()


if __name__ == "__main__":
    main()

"""MAML 元学习训练入口: 产出泛化初值权重 VAE_cold_start.pth。

数据准备:
    1. cut_imgs/ 放通用训练素材 (SAM 切割结果);
    2. datasets/test_dataset.pt 作为特定任务素材 (由 generate_test_dataset.py 生成)。

用法:
    python train_meta.py              # 默认: 100 轮元迭代
"""

import argparse
import os

import torch

from defect_detection.config import CUT_IMAGE_DIR, VAE_COLD_START_CHECKPOINT
from defect_detection.meta_learning import MAML, load_datasets, train_maml_vae, build_meta_loaders


def parse_args():
    parser = argparse.ArgumentParser(description="MAML 元学习训练")
    parser.add_argument("--model", default=str(VAE_COLD_START_CHECKPOINT),
                        help="元学习权重保存路径")
    parser.add_argument("--meta-iterations", type=int, default=100, help="元迭代次数")
    parser.add_argument("--tasks", type=int, default=5, help="每次元更新的任务数")
    parser.add_argument("--inner-steps", type=int, default=5, help="每轮元迭代的内部更新步数")
    parser.add_argument("--samples", type=int, default=10, help="每个任务采样的样本数")
    parser.add_argument("--batch-size", type=int, default=5, help="DataLoader 批大小")
    parser.add_argument("--general-dir", default=str(CUT_IMAGE_DIR), help="通用素材目录")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    general_data, specific_data = load_datasets(general_image_dir=args.general_dir)
    if specific_data is None:
        raise FileNotFoundError(
            "缺少特定任务素材 datasets/test_dataset.pt, 请先运行 generate_test_dataset.py。")

    general_loader, specific_loader = build_meta_loaders(general_data, specific_data,
                                                         batch_size=args.batch_size)

    maml_model = MAML(1e-5, 1e-8, device)

    model_path = args.model
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if os.path.isfile(model_path):
        try:
            maml_model.model.load_state_dict(
                torch.load(model_path, map_location=device, weights_only=True))
            print("Model loaded successfully from '{}'".format(model_path))
        except Exception as e:
            print("Failed to load model. Starting from scratch. Error: ", e)
    else:
        print("No saved model found at '{}'. Starting from scratch.".format(model_path))

    for meta_iteration in range(args.meta_iterations):
        print("-" * 10, "Meta Iteration", meta_iteration, "-" * 10)
        train_maml_vae(maml_model, general_loader, specific_loader, args.tasks,
                       args.inner_steps, args.samples, meta_iteration)

        torch.save(maml_model.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()

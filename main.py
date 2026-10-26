"""缺陷检测推理入口: 对 input/ 目录下的图片批量检测, 结果写入 output/。

用法:
    python main.py                      # 使用默认目录与 saved_model/VAE.pth
    python main.py --no-show            # 不弹窗, 四联对比图保存到 output/
    python main.py --input X --output Y --model Z
"""

import argparse

import torch

from defect_detection.config import INPUT_DIR, OUTPUT_DIR, VAE_CHECKPOINT
from defect_detection.detector import detect_batch


def parse_args():
    parser = argparse.ArgumentParser(description="VAE-GAN 无监督缺陷检测 (推理)")
    parser.add_argument("--input", default=str(INPUT_DIR), help="待检测图片目录")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help="结果输出目录")
    parser.add_argument("--model", default=str(VAE_CHECKPOINT), help="VAE 权重路径")
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto")
    parser.add_argument("--no-show", action="store_true", help="不弹出可视化窗口, 改为保存对比图")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = detect_batch(args.input, args.output, model_path=args.model, device=device,
                         show=not args.no_show)
    print(f"\n完成: 共检测 {len(pairs)} 张图, 结果保存在 {args.output}")


if __name__ == "__main__":
    main()

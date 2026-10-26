"""SAM 批量切割入口: 从 input/ 的每张图中切割最大目标并保存到 output/。

前置条件: saved_model/ 下需要有 sam_vit_l_0b3195.pth (SAM 权重)。

用法:
    python cut_targets.py
    python cut_targets.py --min-area 60000
"""

import argparse

from defect_detection.config import INPUT_DIR, OUTPUT_DIR
from defect_detection.segmentation import cut_targets_in_directory


def parse_args():
    parser = argparse.ArgumentParser(description="SAM 目标批量切割")
    parser.add_argument("--input", default=str(INPUT_DIR), help="原图目录")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help="切割结果输出目录")
    parser.add_argument("--min-area", type=float, default=40000, help="目标掩码的最小面积")
    parser.add_argument("--device", default="cuda", help="推理设备")
    return parser.parse_args()


def main():
    args = parse_args()
    cut_targets_in_directory(args.input, args.output, min_area=args.min_area,
                             device=args.device)


if __name__ == "__main__":
    main()

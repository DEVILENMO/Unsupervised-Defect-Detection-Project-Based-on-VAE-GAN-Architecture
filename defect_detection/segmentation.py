"""基于 Segment Anything (SAM) 的目标切割。

SAM 模型懒加载: 只有真正调用切割函数时才读入 checkpoint,
导入本模块不会有任何模型加载 / 目录扫描等副作用。
"""

import math
import os

import cv2
import numpy as np
from PIL import Image

from defect_detection.config import (
    DISCRIMINATOR_TRAINED_CHECKPOINT,
    OUTPUT_DIR,
    SAM_CHECKPOINT,
    SAM_MODEL_TYPE,
)
from defect_detection.target_discriminator import TargetDiscriminator

try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "segment_anything 未找到, 请在项目根目录运行脚本 "
        "(segment_anything 是随仓库分发的前置依赖)") from e

# SAM 推理前先把图缩到该窗口内, 控制显存与耗时
SAM_WINDOW_MAX_SIZE = (896, 750)

_sam_model = None


def get_sam(device: str = "cuda"):
    """获取 (并按需加载) 共享的 SAM 模型实例。"""
    global _sam_model
    if _sam_model is None:
        if not os.path.isfile(SAM_CHECKPOINT):
            raise FileNotFoundError(
                f"SAM checkpoint 不存在: {SAM_CHECKPOINT}\n"
                "请从 https://github.com/facebookresearch/segment-anything 下载 "
                "sam_vit_l_0b3195.pth 放入 saved_model/ 目录。")
        print(f"Loading SAM ({SAM_MODEL_TYPE}) from {SAM_CHECKPOINT} ...")
        _sam_model = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
    _sam_model.to(device=device)
    return _sam_model


def pad_to_square(arr: np.ndarray):
    """把图像用白边填充成正方形, 返回 (填充后图像, 还原信息)。"""
    h, w = arr.shape[:2]
    max_dim = max(h, w)

    top = (max_dim - h) // 2
    bottom = max_dim - h - top
    left = (max_dim - w) // 2
    right = max_dim - w - left

    padded_arr = np.pad(arr, ((top, bottom), (left, right), (0, 0)),
                        mode="constant", constant_values=255)
    restore_info = {"original_shape": (h, w), "top": top, "left": left}
    return padded_arr, restore_info


def scale_image_to_fit_window(image: np.ndarray, max_width: int = SAM_WINDOW_MAX_SIZE[0],
                              max_height: int = SAM_WINDOW_MAX_SIZE[1]) -> np.ndarray:
    """等比缩放图像, 使其不超过给定窗口大小。"""
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    return cv2.resize(image, None, fx=scale, fy=scale)


def apply_mask(image: np.ndarray, mask: np.ndarray, alpha_channel: bool = True,
               kernel_size=None) -> tuple:
    """用 mask 抠出感兴趣区域。

    Args:
        image: (H, W, 3) BGR/RGB 图像。
        mask: 与图像同尺寸的布尔掩码。
        alpha_channel: True 时把掩码外区域替换为白色 (原 alpha 合成语义); False 时置黑。
        kernel_size: 形如 (a, b) 的二元组时, 先对 mask 做腐蚀/膨胀 + 小连通域过滤再应用。

    Returns:
        (处理后的图像, 实际使用的掩码)
    """
    if kernel_size is not None and len(kernel_size) == 2:
        print("Refining mask...")
        mask_processed = mask.astype(np.uint8) * 255

        kernel_1 = np.ones((kernel_size[0], kernel_size[0]), np.uint8)
        kernel_2 = np.ones((kernel_size[1], kernel_size[1]), np.uint8)

        mask_eroded = cv2.erode(mask_processed, kernel_1, iterations=1)

        # 只保留面积足够大的连通域, 再适度膨胀回去
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_eroded, connectivity=8)
        area_threshold = 10000

        mask_processed = np.zeros_like(mask_eroded)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= area_threshold:
                mask_processed[labels == i] = 255

        mask_dilated = cv2.dilate(mask_processed, kernel_1, iterations=1)
        mask_dilated = cv2.dilate(mask_dilated, kernel_2, iterations=1)
        mask_processed = cv2.erode(mask_dilated, kernel_2, iterations=1)

        mask = mask_processed.astype(bool)

    if alpha_channel:
        # 掩码外区域统一替换为白色 (等价于原 alpha 通道 + 白底合成)
        background = np.where(mask[..., None], image, 255)
        image = background.astype(image.dtype)
    else:
        image = np.where(mask[..., None], image, 0)

    return image, mask


def get_next_filename(base_path, filename: str):
    """生成不冲突的输出文件名: name_1.png, name_2.png, ..."""
    name, ext = os.path.splitext(filename)
    for i in range(1, 101):
        new_name = f"{name}_{i}{ext}"
        if not os.path.exists(os.path.join(base_path, new_name)):
            return new_name
    return None


def save_masked_image(image: np.ndarray, mask: np.ndarray, output_dir, filename: str,
                      crop_mode: bool = True, kernel_size=None):
    """保存掩码抠图结果: 裁剪到掩码包围盒 -> 掩码处理 -> (可选) 填充为正方形。"""
    if crop_mode:
        y, x = np.where(mask)
        y_min, y_max, x_min, x_max = y.min(), y.max(), x.min(), x.max()
        cropped_mask = mask[y_min:y_max + 1, x_min:x_max + 1]
        cropped_image = image[y_min:y_max + 1, x_min:x_max + 1]
        masked_image, _ = apply_mask(cropped_image, cropped_mask, kernel_size=kernel_size)
        masked_image, _ = pad_to_square(masked_image)
    else:
        masked_image, _ = apply_mask(image, mask, kernel_size=kernel_size)

    new_filename = get_next_filename(output_dir, os.path.splitext(filename)[0] + ".png")
    if new_filename is None:
        print("Could not save the image. Too many variations exist.")
        return

    cv2.imwrite(os.path.join(output_dir, new_filename), masked_image,
                [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"Saved as {new_filename}")


def cut_target_from_image(origin_image: Image.Image, reason_mode: str = "cuda",
                          area_lower_limit: float = 30000,
                          area_upper_limit: float = math.inf,
                          pcb_prob: float = 0.2):
    """从原图中切割出目标 (SAM 分割 + 判别器筛选)。

    Args:
        origin_image: PIL 原图。
        reason_mode: 推理设备 ('cuda' / 'cpu')。
        area_lower_limit / area_upper_limit: 目标区域面积过滤范围。
        pcb_prob: 判别器认定为真实目标的概率阈值。

    Returns:
        (抠图结果 PIL 图像, [x_min, x_max, y_min, y_max] 原图坐标, 裁剪后的掩码, None)

    Raises:
        RuntimeError: 没有任何掩码同时满足面积与判别器条件。
    """
    sam = get_sam(device=reason_mode)
    discriminator = get_target_discriminator(reason_mode)

    image_crop = np.array(origin_image)
    image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
    image_scaled = scale_image_to_fit_window(image_rgb)

    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image_scaled)

    masked_image_cut = None
    cropped_mask = None
    coordinates = None

    for mask_info in masks:
        if not (area_lower_limit < mask_info["area"] < area_upper_limit):
            continue

        # 掩码是在缩放后的图上生成的, 先还原到原始分辨率
        target_mask = np.uint8(mask_info["segmentation"]) * 255
        target_mask = cv2.resize(target_mask, (image_crop.shape[1], image_crop.shape[0])) > 0

        y, x = np.where(target_mask)
        y_min, y_max, x_min, x_max = y.min(), y.max(), x.min(), x.max()
        candidate_mask = target_mask[y_min:y_max + 1, x_min:x_max + 1]
        candidate_image = image_crop[y_min:y_max + 1, x_min:x_max + 1]

        candidate_cut, _ = apply_mask(candidate_image, candidate_mask)
        prob = discriminator.predict(Image.fromarray(candidate_cut))
        print(f"The probability of being a real PCB is: {prob:.4f}")

        if prob > pcb_prob:
            # 把包围盒扩成正方形, 便于后续统一 resize
            x_min, x_max, y_min, y_max = _expand_to_square(
                x_min, x_max, y_min, y_max, image_crop.shape[1], image_crop.shape[0])
            cropped_mask = target_mask[y_min:y_max + 1, x_min:x_max + 1]
            cropped_image = image_crop[y_min:y_max + 1, x_min:x_max + 1]
            masked_image_cut, cropped_mask = apply_mask(cropped_image, cropped_mask)
            coordinates = [x_min, x_max, y_min, y_max]
            break

    if masked_image_cut is None:
        raise RuntimeError(
            "No target found: 没有掩码同时满足面积限制与判别器概率阈值, "
            "可尝试调整 area_lower_limit / pcb_prob。")

    return Image.fromarray(masked_image_cut), coordinates, cropped_mask, None


def _expand_to_square(x_min, x_max, y_min, y_max, width, height):
    """把坐标包围盒沿短边扩成正方形, 并裁剪到图像边界内。"""
    if y_max - y_min > x_max - x_min:
        diff = (y_max - y_min) - (x_max - x_min)
        x_min -= diff // 2
        x_max += diff - diff // 2
    elif x_max - x_min > y_max - y_min:
        diff = (x_max - x_min) - (y_max - y_min)
        y_min -= diff // 2
        y_max += diff - diff // 2
    return max(0, x_min), min(width - 1, x_max), max(0, y_min), min(height - 1, y_max)


_discriminator_cache = {}


def get_target_discriminator(device: str = "cuda") -> TargetDiscriminator:
    """获取共享的 TargetDiscriminator 实例 (按设备缓存, 避免每次切割都重新加载权重)。"""
    key = (str(DISCRIMINATOR_TRAINED_CHECKPOINT), str(device))
    if key not in _discriminator_cache:
        _discriminator_cache[key] = TargetDiscriminator(
            DISCRIMINATOR_TRAINED_CHECKPOINT, device=device)
    return _discriminator_cache[key]


def cut_targets_in_directory(input_dir, output_dir=OUTPUT_DIR, min_area: float = 40000,
                             device: str = "cuda"):
    """批量切割: 对目录内每张图选取面积超过阈值的最大掩码并保存抠图结果。"""
    sam = get_sam(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    os.makedirs(output_dir, exist_ok=True)

    filenames = [f for f in os.listdir(input_dir)
                 if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg", ".bmp")]

    for filename in filenames:
        print(f"Processing {filename} ...")
        image = cv2.imread(os.path.join(input_dir, filename))
        if image is None:
            print(f"  无法读取, 跳过: {filename}")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_scaled = scale_image_to_fit_window(image_rgb)
        masks = mask_generator.generate(image_scaled)

        # 取面积超过阈值的最大掩码作为切割目标
        candidates = [m for m in masks if m["area"] > min_area]
        if not candidates:
            print("  未找到满足面积条件的目标区域, 跳过。")
            continue
        best_mask = max(candidates, key=lambda m: m["area"])["segmentation"]

        target_mask = np.uint8(best_mask) * 255
        target_mask = cv2.resize(target_mask, (image.shape[1], image.shape[0])) > 0

        save_masked_image(image, target_mask, output_dir, filename, crop_mode=True)

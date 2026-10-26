"""缺陷检测推理管线: SAM 切割 -> VAE 重建 -> SSIM / 差异分析 -> 缺陷区域定位。"""

import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from defect_detection.config import VAE_CHECKPOINT
from defect_detection.image_utils import convert_image_to_tensor, load_image_with_alpha_channel
from defect_detection.models.vae import build_vae
from defect_detection.segmentation import apply_mask, cut_target_from_image


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算两幅灰度图的结构相似性指数 (SSIM)。"""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """中值滤波去噪。"""
    return cv2.medianBlur(img, 3)


def postprocess_diff(diff_img: np.ndarray) -> np.ndarray:
    """差异图 -> 缺陷二值图: 双阈值 + 形态学平滑 + 连通域筛选。

    只保留在高阈值 (强差异) 处也有响应的连通域, 过滤弱噪声。
    """
    _, high_thresh_bin = cv2.threshold(diff_img, 200, 255, cv2.THRESH_BINARY)
    _, diff_bin = cv2.threshold(diff_img, 45, 255, cv2.THRESH_BINARY)

    diff_bin = cv2.dilate(diff_bin, np.ones((3, 3), np.uint8), iterations=1)
    diff_bin = cv2.erode(diff_bin, np.ones((13, 13), np.uint8), iterations=1)
    diff_bin = cv2.dilate(diff_bin, np.ones((11, 11), np.uint8), iterations=1)

    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(diff_bin, connectivity=8)

    processed = np.zeros_like(diff_bin)
    for i in range(1, num_labels):
        # 注意乘 255 得到 0/255 二值图 (原实现漏乘, 输出 0/1 导致保存的掩码图不可见)
        component_mask = (labels == i).astype(np.uint8) * 255
        if cv2.countNonZero(high_thresh_bin & component_mask) > 0:
            processed |= component_mask
    return processed


def resize_defect_map_to_original_size(defect_map, mask_image_size, coordinates,
                                       original_size) -> np.ndarray:
    """把 512x512 的缺陷图映射回原始大图的对应位置。"""
    origin_cropped = cv2.resize(defect_map, mask_image_size)
    x_min, x_max, y_min, y_max = coordinates
    width, height = original_size

    origin_map = np.zeros((height, width), dtype=np.uint8)
    origin_map[y_min:y_max + 1, x_min:x_max + 1] = origin_cropped
    return origin_map


def detect_defect(file_path: str, vae_model, device="cuda", sam_device=None,
                  show: bool = True, fig_save_path=None):
    """对单张图片做缺陷检测。

    流程: SAM 切割目标 -> VAE 重建 -> 与原图比对 (SSIM + 差异图) -> 定位缺陷区域。

    Args:
        file_path: 输入图片路径。
        vae_model: 已加载权重的 VAEModel。
        device: VAE 推理设备。
        sam_device: SAM 推理设备, 默认与 device 一致。
        show: 是否弹出 matplotlib 可视化窗口。
        fig_save_path: 非空时把可视化四联图保存到该路径。

    Returns:
        (ssim, defect_map): 相似度分数与原尺寸的缺陷二值图 (缺陷处为 255)。
    """
    if sam_device is None:
        sam_device = device

    image = Image.open(file_path)
    print("Prepare to cut PCB from the image...")
    t0 = time.time()

    mask_image, coordinates, mask, _ = cut_target_from_image(image, reason_mode=sam_device)
    x_min, x_max, y_min, y_max = coordinates
    print(f"Cut PCB from image successfully, take {time.time() - t0:.2f} seconds.")

    # ---------------- VAE 重建 ----------------
    vae_model.to(device)
    image_tensor = convert_image_to_tensor(mask_image)
    image_tensor = transforms.Resize((512, 512))(image_tensor)
    input_tensor = image_tensor.to(device).unsqueeze(0)

    t2 = time.time()
    with torch.no_grad():
        z_mu, _ = vae_model.encoder(input_tensor)
        regenerated = vae_model.decoder(z_mu)

    regenerated_img = regenerated.cpu().squeeze(0).permute(1, 2, 0).squeeze().clamp(0, 1)
    print(f"VAE has regenerated the input image, take {time.time() - t2:.2f} seconds.")

    # 重建结果贴回掩码并统一到 512x512
    regenerated_np = (regenerated_img.numpy() * 255).astype(np.uint8)
    regenerated_np = cv2.resize(regenerated_np, (mask.shape[1], mask.shape[0]))
    rgb_image = cv2.cvtColor(regenerated_np, cv2.COLOR_GRAY2RGB)
    gray_regen, _ = apply_mask(rgb_image, mask)
    regenerated_np = cv2.resize(cv2.cvtColor(gray_regen, cv2.COLOR_RGB2GRAY), (512, 512))

    # ---------------- 与原图比对 ----------------
    ori_image = load_image_with_alpha_channel(file_path)
    original_size = ori_image.size
    original_np = np.array(ori_image)
    original_crop, mask = apply_mask(original_np[y_min:y_max + 1, x_min:x_max + 1], mask)
    original_gray = cv2.cvtColor(original_crop, cv2.COLOR_RGB2GRAY)

    original_gray = preprocess_image(cv2.resize(original_gray, (512, 512)))
    reconstructed_gray = preprocess_image(regenerated_np)

    ssim = calculate_ssim(original_gray, reconstructed_gray)
    print("Structural Similarity (SSIM) Index: ", ssim)

    diff_img = np.abs(original_gray.astype(np.float32) - reconstructed_gray.astype(np.float32))
    diff_img = (diff_img * 255.0 / diff_img.max()).astype(np.uint8)
    diff_bin = postprocess_diff(diff_img)

    defect_map = resize_defect_map_to_original_size(diff_bin, mask_image.size, coordinates,
                                                    original_size)
    print(f"Total time cost: {time.time() - t0:.2f} seconds.")

    _visualize(ori_image, reconstructed_gray, diff_img, defect_map, ssim,
               show=show, save_path=fig_save_path)

    return ssim, defect_map


def detect_batch(input_dir, output_dir, model_path=VAE_CHECKPOINT, device="cpu",
                 show: bool = True):
    """批量检测目录下所有图片, 输出缺陷二值图 (反色: 缺陷为黑色) 与 mark.txt 对照表。"""
    device = torch.device(device)
    vae_model = build_vae(model_path, device=device)

    os.makedirs(output_dir, exist_ok=True)

    filenames = sorted(
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg", ".bmp"))

    image_pairs = []
    for i, filename in enumerate(filenames):
        print(f"第{i + 1}张图:", filename)

        # 跳过历史输出与其他处理流程的中间产物
        if "bin" in filename or "_t." in filename:
            print("跳过文件:", filename)
            continue

        fig_save_path = os.path.join(output_dir,
                                     f"{os.path.splitext(filename)[0]}_compare.png")
        score, defect_map = detect_defect(
            os.path.join(input_dir, filename), vae_model, device=device,
            show=show, fig_save_path=None if show else fig_save_path)
        print("相似度:", score)

        defect_map = cv2.bitwise_not(defect_map)
        new_filename = f"{os.path.splitext(filename)[0]}_bin.png"
        cv2.imwrite(os.path.join(output_dir, new_filename), defect_map)

        image_pairs.append(f"{filename} {new_filename}")

    with open(os.path.join(output_dir, "mark.txt"), "w") as f:
        f.write("\n".join(image_pairs))

    return image_pairs


def _visualize(ori_image, reconstructed, diff_img, defect_map, ssim, show=True, save_path=None):
    """绘制 原图/重建/差异/缺陷 四联图。"""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(ori_image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 4, 2)
    plt.imshow(reconstructed, cmap="gray")
    plt.title("Reconstructed Image")
    plt.axis("off")
    plt.subplot(1, 4, 3)
    plt.imshow(diff_img, cmap="jet")
    plt.title("Difference Image")
    plt.axis("off")
    plt.subplot(1, 4, 4)
    plt.imshow(defect_map, cmap="gray")
    plt.title("Defect Regions")
    plt.axis("off")
    plt.suptitle(f"SSIM: {ssim:.4f}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print("Comparison figure saved to", save_path)
    if show:
        plt.show()
    else:
        plt.close()

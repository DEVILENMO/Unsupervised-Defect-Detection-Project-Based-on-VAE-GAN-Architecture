"""图像增强: PIL 级别的几何增强用于扩充训练集, Tensor 级别的退化用于模拟缺陷/噪声。"""

import math
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torchvision import transforms


# ---------------------------------------------------------------------------
# 几何增强 (PIL)
# ---------------------------------------------------------------------------

def rotate(image: Image.Image, angle: float) -> Image.Image:
    """旋转图像 (白色填充) 并裁剪掉白边, 只保留内容区域。"""
    image_array = np.array(image.convert("RGB"))
    height, width = image_array.shape[:2]
    center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image_array, rotation_matrix, (width, height),
                             borderValue=(255, 255, 255))

    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return Image.fromarray(rotated)

    x, y, w, h = cv2.boundingRect(contours[0])
    return Image.fromarray(rotated[y: y + h, x: x + w])


def augment_image_tensor(image_tensor: torch.Tensor, zoom_factors=(), target_size=(512, 512)) -> list:
    """对一张 (C, H, W) 张量做完整的训练集扩充。

    生成内容: 原图 (resize) + 35 次 10 度步进的随机抖动旋转
    + 每个 zoom_factor 的中心裁剪及其 11 次 30 度步进旋转 + 3 次亮度/对比度/饱和度扰动。
    """
    image = transforms.ToPILImage()(image_tensor)
    resize = transforms.Resize(target_size)

    augmented_images = [resize(image_tensor)]

    # 原图 35 次小角度旋转
    for step in range(35):
        rotation = step * 10 + round(random.uniform(-3, 3))
        rotated = rotate(image, rotation).resize(target_size, resample=Image.Resampling.BICUBIC)
        augmented_images.append(transforms.ToTensor()(rotated))

    # 中心裁剪 (zoom) 后各 11 次旋转
    width, height = image.size
    for zoom_factor in zoom_factors:
        crop_size = int(min(width, height) / zoom_factor)
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        cropped = image.crop((left, top, left + crop_size, top + crop_size))
        resized = cropped.resize(target_size, resample=Image.Resampling.BICUBIC)
        augmented_images.append(transforms.ToTensor()(resized))

        for step in range(11):
            rotation = step * 30 + random.uniform(-5, 5)
            rotated = rotate(resized, rotation).resize(target_size,
                                                       resample=Image.Resampling.BICUBIC)
            augmented_images.append(transforms.ToTensor()(rotated))

    # 随机亮度 / 对比度 / 饱和度
    for _ in range(3):
        enhanced = image.copy()
        enhanced = ImageEnhance.Brightness(enhanced).enhance(random.uniform(0.8, 1.2))
        enhanced = ImageEnhance.Contrast(enhanced).enhance(random.uniform(0.8, 1.2))
        enhanced = ImageEnhance.Color(enhanced).enhance(random.uniform(0.8, 1.2))
        augmented_images.append(resize(transforms.ToTensor()(enhanced)))

    return augmented_images


def do_nothing(image_tensor: torch.Tensor, target_size=(512, 512)) -> list:
    """不做任何增强, 只 resize (作为增强关闭时的占位实现)。"""
    return [transforms.Resize(target_size)(image_tensor)]


def limit_size(image_tensor: torch.Tensor, size: int = 512) -> torch.Tensor:
    """统一到 size x size 的单通道灰度张量。"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size, size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    return transform(image_tensor)


def rotate_image(image_tensor: torch.Tensor, angle: int) -> torch.Tensor:
    """旋转 90 度的整数倍, angle 取 0/90/180/270。"""
    rotations = {0: 0, 90: 1, 180: 2, 270: 3}
    if angle not in rotations:
        raise ValueError("Invalid rotation angle. Supported angles are 0, 90, 180, and 270.")
    return torch.rot90(image_tensor, k=rotations[angle], dims=[-2, -1])


# ---------------------------------------------------------------------------
# 退化 / 噪声增强 (Tensor), 训练时用于模拟缺陷掩盖
# ---------------------------------------------------------------------------

def add_stripe_noise_pattern_around_image(image_tensor: torch.Tensor, pattern_width_ratio: float = 0.1,
                                          stripe_scale: float = 0.05, stripe_intensity: float = 0.2):
    """在图像四周叠加水平正弦条纹噪声。"""
    _, H, W = image_tensor.shape
    device = image_tensor.device

    pattern_width = int(min(H, W) * pattern_width_ratio)

    noise_map = torch.zeros((H, W), device=device)
    for i in range(H):
        noise_map[i, :] = torch.sin(torch.tensor(i, device=device) * stripe_scale) * stripe_intensity

    mask = torch.zeros((H, W), device=device)
    mask[:pattern_width, :] = 1
    mask[-pattern_width:, :] = 1
    mask[:, :pattern_width] = 1
    mask[:, -pattern_width:] = 1
    noise_map = noise_map * mask

    return torch.clamp(image_tensor - noise_map, min=0)


def add_gaussian_noise(image_tensor: torch.Tensor, noise_std: float = 0.05):
    """叠加高斯噪声。"""
    noise = torch.randn_like(image_tensor) * noise_std
    return torch.clamp(image_tensor + noise, 0, 1)


def add_salt_pepper_noise(image_tensor: torch.Tensor, noise_density: float = 0.05):
    """叠加椒盐噪声。"""
    num_pixels = int(noise_density * image_tensor.size(1) * image_tensor.size(2))
    coords = torch.randint(0, image_tensor.size(1), (num_pixels, 2), device=image_tensor.device)
    values = torch.randint(0, 2, (num_pixels, image_tensor.size(0)),
                           device=image_tensor.device).float()
    noisy_image = image_tensor.clone()
    noisy_image[:, coords[:, 0], coords[:, 1]] = values.transpose(0, 1)
    return noisy_image


def grid_mask(image_tensor: torch.Tensor, num_blocks_range=(12, 24), block_size_range=(512 // 8, 512 // 5),
              fill_value_range=(0, 0.3)):
    """在图像上按网格随机撒小方块, 模拟块状遮挡缺陷。"""
    image_tensor = image_tensor.clone()
    _, h, w = image_tensor.shape

    num_blocks = torch.randint(num_blocks_range[0], num_blocks_range[1], (1,)).item()
    block_size = torch.randint(block_size_range[0], block_size_range[1], (1,)).item()

    num_rows = math.ceil(math.sqrt(num_blocks))
    num_cols = math.ceil(num_blocks / num_rows)

    grid_size_h = h // num_rows
    grid_size_w = w // num_cols

    for i in range(num_rows):
        for j in range(num_cols):
            if torch.rand(1).item() < 0.5 and (i * num_cols + j) < num_blocks:
                start_h = i * grid_size_h + (grid_size_h - block_size) // 2
                start_w = j * grid_size_w + (grid_size_w - block_size) // 2
                fill_value = (torch.rand(1).item() * (fill_value_range[1] - fill_value_range[0])
                              + fill_value_range[0])
                image_tensor[..., start_h:start_h + block_size, start_w:start_w + block_size] = fill_value

    return image_tensor


def add_gray_stripes(image_tensor: torch.Tensor, num_stripes_range=(1, 3), stripe_width: int = 35,
                     gray_range=(0.3, 0.55)):
    """随机添加灰色长条 (横向 / 纵向 / 45 度 / 30 度 / 60 度), 模拟划痕类缺陷。"""
    _, H, W = image_tensor.shape
    num_stripes = random.randint(num_stripes_range[0], num_stripes_range[1])

    image_with_stripes = image_tensor.clone()

    def fill_square(y_center: float, x_center: float):
        half = stripe_width // 2
        y0, y1 = max(0, int(y_center) - half), min(H, int(y_center) + half)
        x0, x1 = max(0, int(x_center) - half), min(W, int(x_center) + half)
        if y1 > y0 and x1 > x0:
            image_with_stripes[:, y0:y1, x0:x1] = gray_value

    for _ in range(num_stripes):
        direction = random.randint(0, 4)
        gray_value = random.uniform(gray_range[0], gray_range[1])
        half = stripe_width // 2

        if direction == 0:  # 横向: 贯穿整幅图的横条
            y = random.randint(0, H - 1)
            image_with_stripes[:, max(0, y - half):min(H, y + half), :] = gray_value
        elif direction == 1:  # 纵向: 贯穿整幅图的竖条
            x = random.randint(0, W - 1)
            image_with_stripes[:, :, max(0, x - half):min(W, x + half)] = gray_value
        elif direction == 2:  # 45 度
            for i in range(max(-W, -H), min(W, H)):
                if 0 <= i < W and 0 <= H - i - 1 < H:
                    fill_square(H - i - 1, i)
        elif direction == 3:  # 30 度
            for i in range(max(-W, -int(H / math.sqrt(3))), min(W, int(H / math.sqrt(3)))):
                y = int(H - i * math.sqrt(3)) - 1
                if 0 <= i < W and 0 <= y < H:
                    fill_square(y, i)
        else:  # 60 度
            for i in range(max(-W, -int(H * math.sqrt(3))), min(W, int(H * math.sqrt(3)))):
                y = int(H - i / math.sqrt(3)) - 1
                if 0 <= i < W and 0 <= y < H:
                    fill_square(y, i)

    return image_with_stripes


def add_central_distortion(image_tensor: torch.Tensor, distortion_strength_range=(0.2, 0.3),
                           radius_range=(0.35, 0.55)):
    """在图像中心叠加径向扭曲。"""
    device = image_tensor.device
    _, H, W = image_tensor.shape
    center_x, center_y = W / 2, H / 2

    distortion_strength = (torch.rand(1).item()
                           * (distortion_strength_range[1] - distortion_strength_range[0])
                           + distortion_strength_range[0])
    radius = min(W, H) * (torch.rand(1).item() * (radius_range[1] - radius_range[0])
                          + radius_range[0])

    xv, yv = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device),
                            indexing="ij")
    dist = torch.sqrt((xv - center_x) ** 2 + (yv - center_y) ** 2)

    factor = torch.ones_like(dist)
    within_radius = dist < radius
    factor[within_radius] = 1 + distortion_strength * torch.sin(torch.pi * dist[within_radius] / radius)

    xv_new = (xv - center_x) * factor + center_x
    yv_new = (yv - center_y) * factor + center_y

    grid = torch.stack([yv_new / (H - 1) * 2 - 1, xv_new / (W - 1) * 2 - 1], dim=2)

    return F.grid_sample(image_tensor.unsqueeze(0), grid.unsqueeze(0), mode="bilinear",
                         padding_mode="border", align_corners=True).squeeze(0)


def random_kernel_filter(image_tensor: torch.Tensor, kernel_size_range=(5, 11), sigma_range=(0.5, 0.9)):
    """随机大小 / 随机方差的高斯核模糊。"""
    device = image_tensor.device

    kernel_size = torch.randint(kernel_size_range[0] // 2, kernel_size_range[1] // 2 + 1,
                                (1,)).item() * 2 + 1
    sigma = torch.rand(1).item() * (sigma_range[1] - sigma_range[0]) + sigma_range[0]

    kernel = gaussian_kernel(kernel_size, sigma).to(device)
    return F.conv2d(image_tensor.unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0),
                    padding=kernel_size // 2).squeeze(0)


def gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    """生成归一化的 (kernel_size, kernel_size) 高斯卷积核, kernel_size 必须为奇数。"""
    assert kernel_size % 2 == 1, "Kernel size must be odd"

    x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
    y = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
    xy_grid = torch.stack(torch.meshgrid(x, y, indexing="ij"), dim=-1)

    kernel = torch.exp(-torch.sum(xy_grid ** 2, dim=-1) / (2 * sigma ** 2))
    return kernel / kernel.sum()

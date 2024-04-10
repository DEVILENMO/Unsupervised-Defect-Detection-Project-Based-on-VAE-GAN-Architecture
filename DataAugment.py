import math
import random
from math import *

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torchvision import transforms


# image augment


def rotate(img, angle):
    image = np.array(img.convert('RGB'))
    # 获取图像的高度和宽度
    height, width = image.shape[:2]

    # 计算旋转中心点
    center = (width // 2, height // 2)

    # 获取旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 对图像进行旋转,使用白色作为填充颜色
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderValue=(255, 255, 255))

    # 将图像转换为灰度图
    gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)

    # 寻找图像的边界
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 计算图像的边界框
    x, y, w, h = cv2.boundingRect(contours[0])

    # 裁剪图像
    cropped_image = rotated_image[y: y + h, x: x + w]

    return Image.fromarray(cropped_image)


# tensor augment

def augment_image_tensor(image_tensor, zoom_factors=[], target_size=(512, 512)):
    # 转换为PIL图像
    image = transforms.ToPILImage()(image_tensor)

    # 保存所有增强后的图像,先放入原始图像以及镜像后的图像
    augmented_images = [
        transforms.Resize(target_size)(image_tensor),
        # transforms.Resize(target_size)(transforms.ToTensor()(transforms.functional.hflip(image)))
    ]

    rotations = [_ * 10 for _ in range(35)]

    # 对原始图像应用35次10度旋转/镜像
    for rotation in rotations:
        random_deviation = random.uniform(-3, 3)

        # 将随机偏差添加到旋转角度上
        adjusted_rotation = round(rotation + random_deviation)
        rotated_image = rotate(image, adjusted_rotation)
        rotated_image = rotated_image.resize(target_size, resample=Image.Resampling.BICUBIC)

        rotated_tensor = transforms.ToTensor()(rotated_image)
        augmented_images.append(rotated_tensor)
        # augmented_images.append(transforms.ToTensor()(transforms.functional.hflip(rotated_image)))

    # 获取图像尺寸
    width, height = image.size
    # # 对不同倍率裁剪后的图像应用11次30度旋转和镜像
    for zoom_factor in zoom_factors:
        # 计算裁剪区域的坐标
        crop_size = int(min(width, height) / zoom_factor)
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size

        # 裁剪图像并调整大小
        cropped_image = image.crop((left, top, right, bottom))
        resized_image = cropped_image.resize(target_size, resample=Image.Resampling.BICUBIC)

        # 保存裁剪后的图像及其镜像
        resized_tensor = transforms.ToTensor()(resized_image)
        augmented_images.append(resized_tensor)
        # augmented_images.append(transforms.ToTensor()(transforms.functional.hflip(resized_image)))

        # 定义11次30度旋转的变换
        rotations = [_ * 30 for _ in range(11)]

        # 对裁剪后的图像应用11次30度旋转和镜像
        for rotation in rotations:
            # 生成一个-5到5之间的随机偏差
            random_deviation = random.uniform(-5, 5)

            # 将随机偏差添加到旋转角度上
            adjusted_rotation = rotation + random_deviation

            rotated_resized_image = rotate(resized_image, adjusted_rotation)
            rotated_resized_image = rotated_resized_image.resize(target_size, resample=Image.Resampling.BICUBIC)
            # display_image(rotated_resized_image)

            rotated_resized_tensor = transforms.ToTensor()(rotated_resized_image)
            augmented_images.append(rotated_resized_tensor)
            # augmented_images.append(transforms.ToTensor()(transforms.functional.hflip(rotated_resized_image)))

    # 随机调整亮度、对比度和饱和度
    for _ in range(3):
        brightness_factor = random.uniform(0.8, 1.2)
        contrast_factor = random.uniform(0.8, 1.2)
        saturation_factor = random.uniform(0.8, 1.2)

        enhanced_image = image.copy()
        enhancer = ImageEnhance.Brightness(enhanced_image)
        enhanced_image = enhancer.enhance(brightness_factor)
        enhancer = ImageEnhance.Contrast(enhanced_image)
        enhanced_image = enhancer.enhance(contrast_factor)
        enhancer = ImageEnhance.Color(enhanced_image)
        enhanced_image = enhancer.enhance(saturation_factor)

        enhanced_tensor = transforms.Resize(target_size)(transforms.ToTensor()(enhanced_image))
        augmented_images.append(enhanced_tensor)

    # # 随机平移
    # for _ in range(2):
    #     max_translation = 0.1
    #     translate_x = int(random.uniform(-max_translation, max_translation) * width)
    #     translate_y = int(random.uniform(-max_translation, max_translation) * height)
    #     translated_image = transforms.functional.affine(image, 0, (translate_x, translate_y), 1.0, 0)
    #     display_image(translated_image)
    #
    #     translated_tensor = transforms.Resize(target_size)(transforms.ToTensor()(translated_image))
    #     augmented_images.append(translated_tensor)
    #
    # # 随机缩放
    # for _ in range(2):
    #     min_scale = 0.9
    #     max_scale = 1.1
    #     scale_factor = random.uniform(min_scale, max_scale)
    #     scaled_image = transforms.functional.affine(image, 0, (0, 0), scale_factor, 0)
    #
    #     scaled_tensor = transforms.Resize(target_size)(transforms.ToTensor()(scaled_image))
    #     augmented_images.append(scaled_tensor)

    return augmented_images


def do_nothing(image_tensor, target_size=(512, 512)):
    # 保存所有增强后的图像,先放入原始图像以及镜像后的图像
    augmented_images = [
        transforms.Resize(target_size)(image_tensor),
    ]
    return augmented_images


def limit_size(image_tensor):
    # 定义转换链
    transform = transforms.Compose([
        transforms.ToPILImage(),  # 将张量转换为PIL图像
        transforms.Resize((512, 512)),  # 调整图像大小为512x512
        transforms.Grayscale(num_output_channels=1),  # 转换为单通道灰度图像
        transforms.ToTensor()  # 将PIL图像转换回张量
    ])

    # 应用转换链
    processed_tensor = transform(image_tensor)

    return processed_tensor


def add_stripe_noise_pattern_around_image(image_tensor, pattern_width_ratio=0.1, stripe_scale=0.05,
                                          stripe_intensity=0.2):
    C, H, W = image_tensor.shape
    device = image_tensor.device

    pattern_width = int(min(H, W) * pattern_width_ratio)

    # 创建一个条纹状噪声贴图
    noise_map = torch.zeros((H, W), device=device)
    for i in range(H):
        noise_map[i, :] = torch.sin(torch.tensor(i, device=device) * stripe_scale) * stripe_intensity

    # 对噪声贴图进行截取,只保留四周
    mask = torch.zeros((H, W), device=device)
    mask[:pattern_width, :] = 1
    mask[-pattern_width:, :] = 1
    mask[:, :pattern_width] = 1
    mask[:, -pattern_width:] = 1
    noise_map = noise_map * mask

    # 使用广播机制将噪声贴图应用于所有通道
    image_with_pattern = torch.clamp(image_tensor - noise_map, min=0)

    return image_with_pattern


def add_gaussian_noise(image_tensor, noise_std=0.05):
    noise = torch.randn_like(image_tensor) * noise_std
    noisy_image = image_tensor + noise
    noisy_image = torch.clamp(noisy_image, 0, 1)
    return noisy_image


def add_salt_pepper_noise(image_tensor, noise_density=0.05):
    num_pixels = int(noise_density * image_tensor.size(1) * image_tensor.size(2))
    coords = torch.randint(0, image_tensor.size(1), (num_pixels, 2), device=image_tensor.device)
    values = torch.randint(0, 2, (num_pixels, image_tensor.size(0)), device=image_tensor.device).float()
    noisy_image = image_tensor.clone()
    noisy_image[:, coords[:, 0], coords[:, 1]] = values.transpose(0, 1)
    return noisy_image


def grid_mask(image_tensor, num_blocks_range=(12, 24), block_size_range=(512 // 8, 512 // 5),
              fill_value_range=(0, 0.3)):
    """
    在图像上按规律排布随机数量的小方块
    :param image_tensor: 输入图像,PyTorch Tensor,shape 为 (C, H, W)
    :param num_blocks_range: 小方块数量范围,默认为 (12, 24)
    :param block_size_range: 小方块边长范围,默认为 (512//8, 512//5)
    :param fill_value_range: 掩码填充值范围,默认为 (0, 0.3)
    :return: 应用 Grid Mask 后的图像
    """
    image_tensor = image_tensor.clone()
    _, h, w = image_tensor.shape

    num_blocks = torch.randint(num_blocks_range[0], num_blocks_range[1], (1,)).item()
    block_size = torch.randint(block_size_range[0], block_size_range[1], (1,)).item()

    num_rows = ceil(sqrt(num_blocks))
    num_cols = ceil(num_blocks / num_rows)

    grid_size_h = h // num_rows
    grid_size_w = w // num_cols

    for i in range(num_rows):
        for j in range(num_cols):
            if torch.rand(1).item() < 0.5 and (i * num_cols + j) < num_blocks:
                start_h = i * grid_size_h + (grid_size_h - block_size) // 2
                start_w = j * grid_size_w + (grid_size_w - block_size) // 2
                end_h = start_h + block_size
                end_w = start_w + block_size

                # 随机选择填充值
                fill_value = torch.rand(1).item() * (fill_value_range[1] - fill_value_range[0]) + fill_value_range[0]
                image_tensor[..., start_h:end_h, start_w:end_w] = fill_value

    return image_tensor


def add_gray_stripes(image_tensor, num_stripes_range=(1, 3), stripe_width=35, gray_range=(0.3, 0.55)):
    """
    随机向图像中添加灰色长条
    :param image_tensor: 输入图像, PyTorch Tensor, shape 为 (C, H, W)
    :param num_stripes_range: 随机生成的长条数量范围
    :param stripe_width: 长条的宽度(像素)
    :param gray_range: 灰度值的范围,表示为0到1之间的浮点数
    :return: 添加灰色长条后的图像
    """
    C, H, W = image_tensor.shape

    # 随机生成长条的数量
    num_stripes = random.randint(num_stripes_range[0], num_stripes_range[1])

    image_with_stripes = image_tensor.clone()

    for _ in range(num_stripes):
        # 随机决定长条的方向(0为横向,1为纵向,2为45度,3为30度,4为60度)
        direction = random.randint(0, 4)
        # 随机选择灰度值
        gray_value = random.uniform(gray_range[0], gray_range[1])

        if direction == 0:  # 横向长条
            y = random.randint(0, H - 1)
            image_with_stripes[:, max(0, y - stripe_width // 2):min(H, y + stripe_width // 2), :] = gray_value
        elif direction == 1:  # 纵向长条
            x = random.randint(0, W - 1)
            image_with_stripes[:, :, max(0, x - stripe_width // 2):min(W, x + stripe_width // 2)] = gray_value
        elif direction == 2:  # 45度长条
            for i in range(max(-W, -H), min(W, H)):
                if 0 <= i < W and 0 <= H - i - 1 < H:
                    image_with_stripes[:, max(0, H - i - 1 - stripe_width // 2):min(H, H - i - 1 + stripe_width // 2),
                    max(0, i - stripe_width // 2):min(W, i + stripe_width // 2)] = gray_value
        elif direction == 3:  # 30度长条
            for i in range(max(-W, -int(H / math.sqrt(3))), min(W, int(H / math.sqrt(3)))):
                if 0 <= i < W and 0 <= int(H - i * math.sqrt(3)) - 1 < H:
                    image_with_stripes[:, max(0, int(H - i * math.sqrt(3)) - 1 - stripe_width // 2):min(H,
                                                                                                        int(H - i * math.sqrt(
                                                                                                            3)) - 1 + stripe_width // 2),
                    max(0, i - stripe_width // 2):min(W, i + stripe_width // 2)] = gray_value
        else:  # 60度长条
            for i in range(max(-W, -int(H * math.sqrt(3))), min(W, int(H * math.sqrt(3)))):
                if 0 <= i < W and 0 <= int(H - i / math.sqrt(3)) - 1 < H:
                    image_with_stripes[:, max(0, int(H - i / math.sqrt(3)) - 1 - stripe_width // 2):min(H,
                                                                                                        int(H - i / math.sqrt(
                                                                                                            3)) - 1 + stripe_width // 2),
                    max(0, i - stripe_width // 2):min(W, i + stripe_width // 2)] = gray_value

    return image_with_stripes


def add_central_distortion(image_tensor, distortion_strength_range=(0.2, 0.3), radius_range=(0.35, 0.55)):
    """
    在图像中心添加径向扭曲
    :param image_tensor: 输入图像, PyTorch Tensor, shape 为 (C, H, W)
    :param distortion_strength_range: 扭曲强度的随机范围
    :param radius_range: 影响半径的随机范围，以图像宽高的最小值的比例给出
    :return: 添加扭曲后的图像
    """
    # 确保图像在正确的设备上
    device = image_tensor.device

    C, H, W = image_tensor.shape
    center_x, center_y = W / 2, H / 2

    # 随机生成扭曲强度和影响半径
    distortion_strength = torch.rand(1).item() * (distortion_strength_range[1] - distortion_strength_range[0]) + \
                          distortion_strength_range[0]
    radius = torch.rand(1).item() * (radius_range[1] - radius_range[0]) + radius_range[0]
    radius = min(W, H) * radius

    # 生成像素坐标网格
    xv, yv = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing="ij")

    # 计算每个像素距离中心的距离
    dist = torch.sqrt((xv - center_x) ** 2 + (yv - center_y) ** 2)

    # 应用径向扭曲模型
    factor = torch.ones_like(dist)
    within_radius = dist < radius
    factor[within_radius] = 1 + distortion_strength * torch.sin(torch.pi * dist[within_radius] / radius)

    # 计算扭曲后的坐标
    xv_new = (xv - center_x) * factor + center_x
    yv_new = (yv - center_y) * factor + center_y

    # 创建坐标网格以进行采样
    grid = torch.stack([yv_new / (H - 1) * 2 - 1, xv_new / (W - 1) * 2 - 1], dim=2)

    # 重采样图像
    distorted_image = F.grid_sample(image_tensor.unsqueeze(0), grid.unsqueeze(0), mode='bilinear',
                                    padding_mode='border',
                                    align_corners=True).squeeze(0)

    return distorted_image


def random_kernel_filter(image_tensor, kernel_size_range=(5, 11), sigma_range=(0.5, 0.9)):
    """
    对图像应用随机卷积核滤波
    :param image_tensor: 输入图像, PyTorch Tensor, shape 为 (C, H, W)
    :param kernel_size_range: 卷积核大小的随机范围,必须为奇数
    :param sigma_range: 高斯卷积核的标准差随机范围
    :return: 应用卷积核滤波后的图像
    """
    # 确保图像在正确的设备上
    device = image_tensor.device

    # 随机生成卷积核大小和标准差
    kernel_size = torch.randint(kernel_size_range[0] // 2, kernel_size_range[1] // 2 + 1, (1,)).item() * 2 + 1
    sigma = torch.rand(1).item() * (sigma_range[1] - sigma_range[0]) + sigma_range[0]

    # 生成高斯卷积核
    kernel = gaussian_kernel(kernel_size, sigma).to(device)

    # 对图像应用卷积
    filtered_image = F.conv2d(image_tensor.unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0),
                              padding=kernel_size // 2).squeeze(0)

    return filtered_image


def gaussian_kernel(kernel_size, sigma):
    """
    生成高斯卷积核
    :param kernel_size: 卷积核大小,必须为奇数
    :param sigma: 高斯分布的标准差
    :return: 高斯卷积核, PyTorch Tensor, shape 为 (kernel_size, kernel_size)
    """
    assert kernel_size % 2 == 1, "Kernel size must be odd"

    # 生成坐标网格
    x = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32)
    y = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32)
    xy_grid = torch.stack(torch.meshgrid(x, y), dim=-1)

    # 计算高斯分布
    kernel = torch.exp(-torch.sum(xy_grid ** 2, dim=-1) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()

    return kernel


def rotate_image(image_tensor, angle):
    """
    旋转图像
    :param image_tensor: 输入图像,PyTorch Tensor,shape 为 (C, H, W)
    :param angle: 旋转角度,可选值为 0, 90, 180, 270
    :return: 旋转后的图像
    """
    if angle == 0:
        return image_tensor
    elif angle == 90:
        return torch.rot90(image_tensor, k=1, dims=[-2, -1])
    elif angle == 180:
        return torch.rot90(image_tensor, k=2, dims=[-2, -1])
    elif angle == 270:
        return torch.rot90(image_tensor, k=3, dims=[-2, -1])
    else:
        raise ValueError("Invalid rotation angle. Supported angles are 0, 90, 180, and 270.")

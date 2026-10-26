"""图像读写与预处理工具 (PIL / OpenCV / Tensor 之间的转换)。"""

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def composite_on_white(image: Image.Image) -> Image.Image:
    """把 RGBA 图像合成到白色背景上, 透明区域变成白色。"""
    background = Image.new("RGBA", image.size, (255, 255, 255))
    return Image.alpha_composite(background, image)


def load_image_with_alpha_channel(image_path: str) -> Image.Image:
    """读取图片并保留 alpha 通道语义: 返回合成白底后的 RGBA 图像。"""
    return composite_on_white(Image.open(image_path).convert("RGBA"))


def load_image_to_tensor(image_path: str) -> torch.Tensor:
    """读取图片 -> 白底灰度图 -> (1, H, W) 张量, 取值 [0, 1]。"""
    gray = composite_on_white(Image.open(image_path).convert("RGBA")).convert("L")
    return transforms.ToTensor()(gray)


def convert_image_to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL 图像 -> 白底灰度图 -> (1, H, W) 张量。"""
    gray = composite_on_white(image.convert("RGBA")).convert("L")
    return transforms.ToTensor()(gray)


def convert_to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def display_image(image):
    """用 matplotlib 显示图像 (调试用)。"""
    import matplotlib.pyplot as plt

    plt.imshow(image, cmap="gray" if image.ndim == 2 else None)
    plt.axis("off")
    plt.show()


# ---------------------------------------------------------------------------
# OpenCV 侧工具
# ---------------------------------------------------------------------------

def read_img(image_path: str):
    """读取图像, 失败时返回 None。"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found or invalid image format: {image_path}")
    return image


def show_img(images, win_name: str = None):
    """在一个 1280x720 窗口内显示一张或多张图像 (调试用)。"""
    if not isinstance(images, list):
        images = [images]

    for i, image in enumerate(images):
        screen_res = 1280, 720
        scale = min(screen_res[0] / image.shape[1], screen_res[1] / image.shape[0])
        window_width = int(image.shape[1] * scale)
        window_height = int(image.shape[0] * scale)

        window_name = f"image {i}" if win_name is None else f"{win_name} {i}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, window_width, window_height)
        cv2.imshow(window_name, image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_pre_processing_gray(image: np.ndarray) -> np.ndarray:
    """BGR 图像转灰度。"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def img_pre_processing_binary(image: np.ndarray) -> np.ndarray:
    """BGR 图像自适应二值化 + Otsu 二次二值化 + 形态学平滑 + 高斯模糊。"""
    b_channel, g_channel, r_channel = cv2.split(image)

    binary_channels = []
    for channel in [b_channel, g_channel, r_channel]:
        blurred_channel = cv2.medianBlur(channel, 9)
        binary_channel = cv2.adaptiveThreshold(
            blurred_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 6)
        binary_channels.append(binary_channel)

    binary_image = cv2.merge(binary_channels)
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(binary_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    binary_image = cv2.erode(binary_image, kernel, iterations=1)
    binary_image = cv2.dilate(binary_image, kernel, iterations=1)
    binary_image = cv2.GaussianBlur(binary_image, (5, 5), 0)
    return binary_image

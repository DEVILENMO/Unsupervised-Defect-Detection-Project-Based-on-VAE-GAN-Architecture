import cv2
import numpy as np


def read_img(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found or invalid image format")
        return None
    return image


def show_img(images, win_name=None):
    if not isinstance(images, list):
        images = [images]

    for i, image in enumerate(images):
        # 调整图像大小以适应屏幕
        screen_res = 1280, 720  # 示例屏幕分辨率
        scale_width = screen_res[0] / image.shape[1]
        scale_height = screen_res[1] / image.shape[0]
        scale = min(scale_width, scale_height)

        window_width = int(image.shape[1] * scale)
        window_height = int(image.shape[0] * scale)

        # 为每个图像创建一个唯一的窗口
        if win_name is None:
            window_name = f'image {i}'
        else:
            window_name = win_name + f' {i}'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, window_width, window_height)
        cv2.imshow(window_name, image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_pre_processing_gray(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


def img_pre_processing_binary(image):
    # 分离 BGR 通道
    b_channel, g_channel, r_channel = cv2.split(image)

    # 对每个通道分别应用中值模糊和自适应阈值二值化
    binary_channels = []
    for channel in [b_channel, g_channel, r_channel]:
        blurred_channel = cv2.medianBlur(channel, 9)
        binary_channel = cv2.adaptiveThreshold(blurred_channel, 255,
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 17, 6)
        binary_channels.append(binary_channel)

    # 合并二值化后的通道
    binary_image = cv2.merge(binary_channels)
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(binary_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    binary_image = cv2.erode(binary_image, kernel, iterations=1)
    binary_image = cv2.dilate(binary_image, kernel, iterations=1)

    # 应用高斯模糊到合并后的二值化图像
    binary_image = cv2.GaussianBlur(binary_image, (5, 5), 0)
    return binary_image

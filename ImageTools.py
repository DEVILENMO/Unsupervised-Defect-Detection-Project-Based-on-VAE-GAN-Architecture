from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms


def load_image_to_tensor(image_path: str):
    image = Image.open(image_path).convert("RGBA")  # 读取图像并转换为RGB模式

    # 如果图像有alpha通道,将RGBA图像转换为灰度图,并将透明部分填充为白色
    background = Image.new("RGBA", image.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image)
    gray_image = alpha_composite.convert("L")

    # 显示灰度图像
    # plt.figure(figsize=(8, 8))
    # plt.imshow(gray_image, cmap='gray')
    # plt.axis('off')
    # plt.show()

    transform = transforms.Compose([
        transforms.ToTensor()  # 将图像转换为PyTorch张量
    ])

    image_tensor = transform(gray_image)  # 将灰度图像转换为张量
    return image_tensor


def load_image_with_alpha_channel(image_path: str):
    image = Image.open(image_path).convert("RGBA")  # 读取图像并转换为RGB模式

    # 如果图像有alpha通道,将RGBA图像转换为灰度图,并将透明部分填充为白色
    background = Image.new("RGBA", image.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image)
    # gray_image = alpha_composite.convert("L")

    # return gray_image
    return alpha_composite


def convert_image_to_tensor(image_pcb):
    image = image_pcb.convert("RGBA")  # 读取图像并转换为RGB模式

    # 如果图像有alpha通道,将RGBA图像转换为灰度图,并将透明部分填充为白色
    background = Image.new("RGBA", image.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image)
    gray_image = alpha_composite.convert("L")

    # 显示灰度图像
    # plt.figure(figsize=(8, 8))
    # plt.imshow(gray_image, cmap='gray')
    # plt.axis('off')
    # plt.show()

    transform = transforms.Compose([
        transforms.ToTensor()  # 将图像转换为PyTorch张量
    ])

    image_tensor = transform(gray_image)  # 将灰度图像转换为张量
    return image_tensor


def display_image(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def convert_to_rgb(img):
    return img.convert('RGB')

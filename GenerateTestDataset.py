import os

import torch

from ImageTools import *


def load_test_dataset(test_imgs_dir='./test_imgs/'):
    # 检查测试图片目录是否存在
    if not os.path.exists(test_imgs_dir):
        print(f"Test images directory not found: {test_imgs_dir}")
        return None

    # 获取目录下所有图片文件的路径
    img_paths = [os.path.join(test_imgs_dir, f) for f in os.listdir(test_imgs_dir) if
                 f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    # 定义图片预处理转换
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # 调整图片大小为512x512
        transforms.ToTensor(),  # 将图片转换为张量
    ])

    # 加载图片并应用预处理转换
    test_dataset = []
    for img_path in img_paths:
        image = Image.open(img_path).convert("RGBA")  # 读取图像并转换为RGBA模式

        # 将RGBA图像转换为灰度图,并将透明部分填充为白色
        background = Image.new("RGBA", image.size, (255, 255, 255))
        alpha_composite = Image.alpha_composite(background, image)
        gray_image = alpha_composite.convert("L")
        image_tensor = transform(gray_image)  # 将灰度图像转换为张量
        test_dataset.append(image_tensor)

    # 将图片数据转换为张量
    test_dataset = torch.stack(test_dataset)
    print(f'Test datas have been loaded from {test_imgs_dir}')
    return test_dataset


if __name__ == '__main__':
    # 调用函数加载测试图片数据集
    test_dataset = load_test_dataset()

    if test_dataset is not None:
        print(f"Test dataset loaded successfully. Shape: {test_dataset.shape}")
        # 可以将test_dataset保存到文件中，类似于提供的代码
        test_dataset_path = './datasets/test_dataset.pt'
        torch.save(test_dataset, test_dataset_path)
        print(f"Test dataset has been saved to {test_dataset_path}.")
    else:
        print("Failed to load test dataset.")

"""数据集的构建与加载。"""

import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from defect_detection.config import IMAGE_EXTENSIONS, TEST_IMAGE_DIR
from defect_detection.data.augment import augment_image_tensor, do_nothing, limit_size
from defect_detection.image_utils import composite_on_white


class TensorImageDataset(Dataset):
    """持有已加载张量的数据集, __getitem__ 直接返回 (C, H, W) 张量。"""

    def __init__(self, images, transform=None):
        """
        Args:
            images: 张量列表 (或可索引的张量集合)。
            transform: 可选变换 (当前流程中张量已完成预处理, 通常为 None)。
        """
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


def list_image_files(directory) -> list:
    """列出目录下所有受支持的图片文件, 按文件名排序。"""
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )


def load_image_datasets(image_dir) -> TensorImageDataset:
    """读取目录下全部图片, 转为 (1, H, W) 灰度张量数据集。"""
    image_paths = list_image_files(image_dir)
    images = [load_image_to_tensor(path) for path in image_paths]
    print(f"Loaded {len(images)} images from '{image_dir}'.")
    return TensorImageDataset(images)


def load_test_dataset(test_image_dir=TEST_IMAGE_DIR) -> torch.Tensor:
    """读取测试图片目录, 返回 (N, 1, 512, 512) 的张量 (用于训练时的验证集)。"""
    if not os.path.isdir(test_image_dir):
        print(f"Test images directory not found: {test_image_dir}")
        return None

    image_paths = list_image_files(test_image_dir)
    if not image_paths:
        print(f"No test images found in: {test_image_dir}")
        return None

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    tensors = []
    for path in image_paths:
        # 白底合成 + 转灰度, 处理 SAM 切割产物自带的 alpha 通道
        gray = composite_on_white(Image.open(path).convert("RGBA")).convert("L")
        tensors.append(transform(gray))

    test_dataset = torch.stack(tensors)
    print(f"Test data loaded from {test_image_dir}: shape {tuple(test_dataset.shape)}")
    return test_dataset


def augment_tensor_dataset(tensor_dataset, augment_fn=augment_image_tensor,
                           target_size=(512, 512)) -> torch.Tensor:
    """对数据集中每张张量应用增强函数, 再统一为 512x512 灰度张量并堆叠。

    Args:
        tensor_dataset: 可迭代的张量集合 (TensorImageDataset / ConcatDataset / 张量)。
        augment_fn: 单张图像的增强函数, 返回张量列表; 传入 do_nothing 可关闭增强。
    """
    augmented_tensors = []
    for tensor in tensor_dataset:
        for augmented in augment_fn(tensor):
            augmented_tensors.append(limit_size(augmented))
    return torch.stack(augmented_tensors)

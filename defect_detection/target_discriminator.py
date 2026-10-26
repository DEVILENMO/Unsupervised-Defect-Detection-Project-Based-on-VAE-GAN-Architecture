"""目标判别器: 用训练好的 Discriminator 判断 SAM 抠出的图像是否为真实目标 (如 PCB)。"""

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from defect_detection.image_utils import convert_image_to_tensor, load_image_to_tensor
from defect_detection.models.discriminator import Discriminator


class TargetDiscriminator:
    def __init__(self, model_path, device: str = "cuda"):
        self.device = device
        self.model = Discriminator().to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        self.transform = transforms.Resize((512, 512))

    def predict(self, image_pcb: Image.Image) -> float:
        """对单张 PIL 图像返回真实目标概率。"""
        image = convert_image_to_tensor(image_pcb)
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(image).item()

    def predict_batch(self, image_path_list) -> np.ndarray:
        """对一组图片路径批量预测, 返回概率数组。"""
        images = [self.transform(load_image_to_tensor(path)) for path in image_path_list]
        images = torch.stack(images).to(self.device)
        with torch.no_grad():
            return self.model(images).cpu().numpy()

import os

import torch

from ImageTools import *
from GAN import Discriminator


class TargetDiscriminator:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = Discriminator().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Resize((512, 512))

    def predict(self, image_pcb):
        # image = load_image_to_tensor(image_path)
        image = convert_image_to_tensor(image_pcb)
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prob = self.model(image).item()

        return prob

    def predict_batch(self, image_path_list):
        images = []
        for image_path in image_path_list:
            image = load_image_to_tensor(image_path)
            image = self.transform(image)
            images.append(image)

        images = torch.stack(images).to(self.device)

        with torch.no_grad():
            probs = self.model(images).cpu().numpy()

        return probs


if __name__ == '__main__':
    discriminator = TargetDiscriminator('saved_model/Discriminator_trained.pth')

    # 读取图片
    image_directory = 'dis_test/'
    image_paths = [os.path.join(image_directory, filename) for filename in os.listdir(image_directory) if filename.endswith(('.jpg', '.jpeg', '.png'))]

    # 批量图片预测
    probs = discriminator.predict_batch(image_paths)
    for path, prob in zip(image_paths, probs):
        # 获取图片名称
        image_name = os.path.basename(path)
        print(f'The probability of {image_name} being a real PCB is: {prob:.4f}')

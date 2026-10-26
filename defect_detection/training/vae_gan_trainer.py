"""VAE-GAN 训练器 (本项目的主训练流程)。

训练方式 (与论文实现一致, 勿随意改动损失权重):
- 生成器是 VAE: 输入加噪/退化的 x_ -> 重建 y; y 再过一遍自身得到 z (修改版的循环重建);
- 损失 = 重建损失 * recon_weight + KL 散度 * kl_weight + 对抗损失 * adv_weight + 二次重建 * regen_weight;
- 判别器是 PatchGAN, 用 LSGAN 风度的 MSE 损失分别对 (真实 x, 生成 y, 生成 z) 更新;
- 训练时的随机退化 (方块遮挡 / 灰条纹 / 各类噪声) 用于模拟缺陷, 提升缺陷区域的重建能力。
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torchvision.utils import save_image

from defect_detection.config import (
    IMAGE_SIZE,
    INPUT_CHANNEL,
    LATENT_DIM,
    REGEN_VIEW_DIR,
    TRAIN_RESULT_DIR,
)
from defect_detection.data.augment import (
    add_central_distortion,
    add_gaussian_noise,
    add_gray_stripes,
    add_salt_pepper_noise,
    add_stripe_noise_pattern_around_image,
    grid_mask,
    random_kernel_filter,
    rotate_image,
)
from defect_detection.image_utils import convert_to_rgb, img_pre_processing_gray
from defect_detection.models.discriminator import Discriminator
from defect_detection.models.vae import VAEDecoder, VAEEncoder, VAEModel
from defect_detection.training.base import ModelLoader

# 损失权重 (与已训练权重对应, 调整前请先了解其对训练稳定性的影响)
adv_weight = 900
recon_weight = 0.4
kl_weight = 1.0
regen_weight = 0.6


class VAEGANModelLoader(ModelLoader):
    def __init__(self, train_dataset, test_dataset, batch_size, model_path: str,
                 discriminator_path: str, if_early_stop: bool = False,
                 debug_mode: bool = False):
        super().__init__(train_dataset, test_dataset, batch_size, model_path,
                         if_early_stop, debug_mode)
        print("-" * 10, "Loading VAE-GAN model", "-" * 10)

        self.latent_dim = LATENT_DIM
        encoder = VAEEncoder(self.latent_dim)
        decoder = VAEDecoder(self.latent_dim)
        self.model = VAEModel(encoder, decoder).to(self.device)

        self.lr = 1e-4
        self.G_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.G_scheduler = CosineAnnealingLR(self.G_optimizer, T_max=20, eta_min=1e-5)

        self.discriminator = Discriminator(INPUT_CHANNEL).to(self.device)
        self.D_lr = 1e-8
        self.D_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.D_lr)
        self.D_scheduler = CosineAnnealingLR(self.D_optimizer, T_max=20, eta_min=0)

        # LSGAN 风格的对抗损失
        self.criterion_GAN = nn.MSELoss()

        self.discriminator_path = discriminator_path
        self.real_labels = torch.ones(batch_size).to(self.device)
        self.fake_labels = torch.zeros(batch_size).to(self.device)

        self.load_model()

    def load_model(self):
        model_dir = os.path.dirname(self.model_path)
        print("Try to load model from", self.model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            print(f"Created directory '{model_dir}' for saving models.")

        if os.path.isfile(self.model_path):
            try:
                self.model.load_state_dict(
                    torch.load(self.model_path, map_location=self.device, weights_only=True))
                print("VAE model loaded successfully from '{}'".format(self.model_path))
            except Exception as e:
                print("Failed to load VAE model. Starting from scratch. Error: ", e)
        else:
            print("No saved model found at '{}'. Starting from scratch.".format(self.model_path))

        if os.path.isfile(self.discriminator_path):
            try:
                self.discriminator.load_state_dict(
                    torch.load(self.discriminator_path, map_location=self.device,
                               weights_only=True))
                print("Discriminator model loaded successfully from '{}'".format(
                    self.discriminator_path))
            except Exception as e:
                print("Failed to load discriminator model. Starting from scratch. Error: ", e)
        else:
            print("No saved model found at '{}'. Starting from scratch.".format(
                self.discriminator_path))

    def save_dis_model(self):
        """保存判别器权重。"""
        torch.save(self.discriminator.state_dict(), self.discriminator_path)
        print(f"Discriminator model saved to {self.discriminator_path}")

    # ------------------------------------------------------------ 训练时随机退化

    @staticmethod
    def _apply_random_degradation(batch: torch.Tensor) -> torch.Tensor:
        """对一批图像先随机旋转 90 度倍数, 再逐张叠加随机退化, 模拟缺陷/噪声。"""
        batch = batch.clone()
        rotation_angles = torch.randint(0, 4, (batch.size(0),)) * 90
        for i in range(batch.size(0)):
            batch[i] = rotate_image(batch[i], rotation_angles[i].item())

        for idx in range(batch.size(0)):
            image = batch[idx]

            # 第一阶段: 块状遮挡 / 灰条纹 (各 25% 概率)
            r = random.random()
            if r <= 0.25:
                image = grid_mask(image)
            elif r <= 0.5:
                image = add_gray_stripes(image)

            # 第二阶段: 噪声 / 扭曲 / 模糊 (共 90% 概率)
            r = random.random()
            if r <= 0.15:
                image = add_salt_pepper_noise(image)
            elif r <= 0.3:
                image = add_gaussian_noise(image)
            elif r <= 0.5:
                image = add_central_distortion(image)
            elif r <= 0.7:
                image = random_kernel_filter(image)
            elif r <= 0.9:
                image = add_stripe_noise_pattern_around_image(image)

            batch[idx] = image

        return batch

    # ------------------------------------------------------------ 训练

    def _test_epoch_vae(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, x in enumerate(self.test_iterator):
                x = x.view(-1, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE).to(self.device)

                x_sample, z_mu, z_var = self.model(x)
                recon_loss = F.mse_loss(x_sample, x, reduction="sum")
                kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1.0 - z_var)

                g_loss = recon_loss * recon_weight + kl_loss * kl_weight
                test_loss += g_loss.item()

                print(f"VAE test batch {i}, g_loss: {g_loss.item() / self.batch_size}")

        return test_loss / len(self.test_dataset)

    def train(self, epochs: int = 100, test_interval: int = 10):
        train_losses_G = []
        train_losses_D = []
        test_losses_G = []

        for epoch in range(epochs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            train_loss_G = 0
            train_loss_D = 0

            for batch, x in enumerate(self.train_iterator):
                x = x.view(-1, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE).to(self.device)
                x_ = self._apply_random_degradation(x)

                # ---------------- 生成器训练: x_ -> y, y -> z ----------------
                self.G_optimizer.zero_grad()

                y, z_mu_y, z_var_y = self.model(x_)
                z, z_mu_z, z_var_z = self.model(y)

                recon_loss_y = F.mse_loss(y, x, reduction="sum")
                kl_loss_y = 0.5 * torch.sum(torch.exp(z_var_y) + z_mu_y ** 2 - 1.0 - z_var_y)
                regen_loss = F.mse_loss(x, z, reduction="sum")
                kl_loss_z = 0.5 * torch.sum(torch.exp(z_var_z) + z_mu_z ** 2 - 1.0 - z_var_z)

                loss_G_adv = (self.criterion_GAN(self.discriminator(y), self.real_labels)
                              + self.criterion_GAN(self.discriminator(z), self.real_labels))

                loss_G = (recon_loss_y * recon_weight
                          + (kl_loss_y + kl_loss_z) * kl_weight
                          + loss_G_adv * adv_weight
                          + regen_loss * regen_weight)
                loss_G.backward()
                self.G_optimizer.step()

                # ---------------- 判别器训练: 真实 x vs 生成 y / z ----------------
                self.D_optimizer.zero_grad()
                real_loss = self.criterion_GAN(self.discriminator(x), self.real_labels)
                fake_loss_y = self.criterion_GAN(self.discriminator(y.detach()), self.fake_labels)
                fake_loss_z = self.criterion_GAN(self.discriminator(z.detach()), self.fake_labels)
                loss_D = (real_loss + fake_loss_y + fake_loss_z) / 3
                loss_D.backward()
                self.D_optimizer.step()

                train_loss_G += loss_G.item()
                train_loss_D += loss_D.item()

                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch + 1}/{len(self.train_iterator)}], "
                      f"D Loss: {loss_D.item():.4f}, "
                      f"G Loss: {loss_G.item() / self.batch_size:.4f}")

                # 注意: 原实现即按 batch 步进调度器 (T_max=20), 保持了原有训练动态
                self.G_scheduler.step()
                self.D_scheduler.step()

            train_loss_G /= len(self.train_dataset)
            train_loss_D /= len(self.train_dataset)
            train_losses_G.append(train_loss_G)
            train_losses_D.append(train_loss_D)

            test_loss_g = self._test_epoch_vae()
            test_losses_G.append(test_loss_g)

            print(f"Epoch {epoch + 1}, Train G Loss: {train_loss_G:.6f}, "
                  f"Train D Loss: {train_loss_D:.6f}, Test G Loss: {test_loss_g}")

            self.save_model()
            self.save_dis_model()

            if (epoch + 1) % test_interval == 0:
                self._regenerate_view_images(prefix=f"epoch_{epoch + 1}")

        with open("cycle_gan_losses.txt", "w") as f:
            f.write("Epoch\tTrain_G_Loss\tTrain_D_Loss\tTest_G_Loss\n")
            for e in range(epochs):
                f.write(f"{e}\t{train_losses_G[e]:.6f}\t{train_losses_D[e]:.6f}\t"
                        f"{test_losses_G[e]}\n")

    # ------------------------------------------------------------ 重建可视化

    def _regenerate_view_images(self, prefix: str):
        """对 REGEN_VIEW_DIR 中的 bmp/png 图片做重建并保存, 用于训练过程可视化。"""
        if not os.path.isdir(REGEN_VIEW_DIR):
            return
        view_files = [f for f in os.listdir(REGEN_VIEW_DIR)
                      if os.path.splitext(f)[1].lower() in (".bmp", ".png")]
        for i, filename in enumerate(view_files):
            self.regenerate_test(Image.open(os.path.join(REGEN_VIEW_DIR, filename)),
                                 f"{prefix}_test_img_{i + 1}")

    def regenerate_test(self, input_image: Image.Image, file_name: str):
        """编码 -> 取潜变量均值 -> 解码, 保存重建结果。"""
        print("-" * 3, "regenerate_test", "-" * 3)
        os.makedirs(TRAIN_RESULT_DIR, exist_ok=True)

        if INPUT_CHANNEL == 3:
            input_image = convert_to_rgb(input_image)
        else:
            cv_image = np.array(input_image.convert("RGB"))[:, :, ::-1]
            cv_image = img_pre_processing_gray(cv_image)
            input_image = Image.fromarray(cv_image)

        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
        input_tensor = transform(input_image).to(self.device).unsqueeze(0)

        with torch.no_grad():
            z_mu, _ = self.model.encoder(input_tensor)
            regenerated = self.model.decoder(z_mu)

        filename = TRAIN_RESULT_DIR / f"{file_name}.png"
        save_image(regenerated.cpu(), filename, normalize=True)
        print("result saved to", filename)

    def test(self):
        self._regenerate_view_images(prefix="test")

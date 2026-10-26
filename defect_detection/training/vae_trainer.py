"""纯 VAE 训练器 (不带对抗损失), 用于 VAE 单独预训练。"""

import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms

from defect_detection.config import (
    IMAGE_SIZE,
    INPUT_CHANNEL,
    LATENT_DIM,
    REGEN_VIEW_DIR,
    VAE_TEST_DIR,
)
from defect_detection.image_utils import convert_to_rgb, img_pre_processing_gray
from defect_detection.models.vae import VAEDecoder, VAEEncoder, VAEModel
from defect_detection.training.base import ModelLoader

import numpy as np
import cv2


class VAEModelLoader(ModelLoader):
    def __init__(self, train_dataset, test_dataset, batch_size, model_path: str,
                 if_early_stop: bool = False, debug_mode: bool = False):
        super().__init__(train_dataset, test_dataset, batch_size, model_path,
                         if_early_stop, debug_mode)
        print("-" * 10, "Loading VAE model", "-" * 10)

        self.latent_dim = LATENT_DIM
        encoder = VAEEncoder(self.latent_dim)
        decoder = VAEDecoder(self.latent_dim)
        self.model = VAEModel(encoder, decoder).to(self.device)

        self.lr = 1e-4
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-5)

        self.load_model()
        self.train_losses = []
        self.test_losses = []

    # ------------------------------------------------------------ 训练

    def _train_epoch(self):
        self.model.train()
        train_loss = 0
        for i, x in enumerate(self.train_iterator):
            x = x.view(-1, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE).to(self.device)

            self.optimizer.zero_grad()
            x_sample, z_mu, z_var = self.model(x)

            recon_loss = F.mse_loss(x_sample, x, reduction="sum")
            kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1.0 - z_var)
            loss = recon_loss + kl_loss

            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

            print(f"Train batch {i}, loss: {loss.item() / self.batch_size}")

        return train_loss / len(self.train_dataset)

    def _test_epoch(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, x in enumerate(self.test_iterator):
                x = x.view(-1, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE).to(self.device)

                x_sample, z_mu, z_var = self.model(x)
                recon_loss = F.mse_loss(x_sample, x, reduction="sum")
                kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1.0 - z_var)
                loss = recon_loss + kl_loss
                test_loss += loss.item()

                print(f"Test batch {i}, loss: {loss.item() / self.batch_size}")

        return test_loss / len(self.test_dataset)

    def train(self, epochs: int = 50, test_interval: int = 10):
        if self.if_early_stop:
            best_test_loss = float("inf")
            patience_counter = 0

        for e in range(epochs):
            print("-" * 10, "Train epoch", e, "started!", "-" * 10)
            train_loss = self._train_epoch()
            test_loss = self._test_epoch()
            print(f"Epoch {e}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}")

            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)

            if self.scheduler:
                self.scheduler.step()
            self.save_model(test_loss)

            if (e + 1) % test_interval == 0:
                self._regenerate_view_images(prefix=f"epoch_{e + 1}")

            if self.if_early_stop:
                if best_test_loss > test_loss:
                    best_test_loss = test_loss
                    patience_counter = 1
                else:
                    patience_counter += 1
                if patience_counter > max(epochs / 5, 10):
                    print("Training interrupted to avoid overfitting.")
                    break

        print(f"Final Train Loss: {self.train_losses[-1]:.2f}, "
              f"Final Test Loss: {self.test_losses[-1]:.2f}")
        self._save_losses_log()

    def _save_losses_log(self, log_path="losses.txt"):
        with open(log_path, "w") as f:
            f.write("Train Losses:\n")
            f.write(", ".join(map(str, self.train_losses)))
            f.write("\n\nTest Losses:\n")
            f.write(", ".join(map(str, self.test_losses)))

    # ------------------------------------------------------------ 重建可视化

    def _regenerate_view_images(self, prefix: str):
        """对 REGEN_VIEW_DIR 中的 bmp 图片做重建并保存, 用于训练过程可视化。"""
        if not os.path.isdir(REGEN_VIEW_DIR):
            return
        view_files = [f for f in os.listdir(REGEN_VIEW_DIR)
                      if os.path.splitext(f)[1].lower() == ".bmp"]
        for i, filename in enumerate(view_files):
            self.regenerate_test(Image.open(os.path.join(REGEN_VIEW_DIR, filename)),
                                 f"{prefix}_test_img_{i + 1}")

    def random_generate_test(self, test_time: int = 10, picture_name: str = "test_image"):
        """从先验分布随机采样潜向量并解码, 检查生成器状态。"""
        print("-" * 3, "random_generate_test", "-" * 3)
        os.makedirs(VAE_TEST_DIR, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i in range(test_time):
                z = torch.randn(1, self.latent_dim).to(self.device)
                reconstructed = self.model.decoder(z)
                img = reconstructed.cpu().squeeze(0).permute(1, 2, 0)
                img = (img.numpy() * 255).astype(np.uint8).squeeze()

                filename = VAE_TEST_DIR / f"{picture_name}_{i}.png"
                Image.fromarray(img).save(filename)
                print("result saved to", filename)

    def regenerate_test(self, input_image: Image.Image, file_name: str):
        """编码 -> 取潜变量均值 -> 解码, 保存重建结果。"""
        print("-" * 3, "regenerate_test", "-" * 3)
        os.makedirs(VAE_TEST_DIR, exist_ok=True)

        if INPUT_CHANNEL == 3:
            input_image = convert_to_rgb(input_image)
            transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
            ])
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

        img = regenerated.cpu().squeeze(0).permute(1, 2, 0)
        img_np = img.numpy()
        if img_np.dtype in (np.float32, np.float64):
            img_np = (img_np * 255).astype(np.uint8)

        filename = VAE_TEST_DIR / f"{file_name}.png"
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            Image.fromarray(img_np).save(filename)
        else:
            Image.fromarray(img_np.squeeze(), mode="L").save(filename)
        print("result saved to", filename)

    def test(self):
        self._regenerate_view_images(prefix="test")

import random
import math
from math import sqrt, ceil

import matplotlib.pyplot as plt
from torchvision.utils import save_image

from AutoEncoder import *
from DataLoader import convert_to_rgb
from ImagePreProcessing import img_pre_processing_gray
from ModelLoader import ModelLoader
from DataAugment import *

adv_weight = 900
recon_weight = 0.4
kl_weight = 1.0
regen_weight = 0.6


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc=input_channel, ndf=64):
        super(Discriminator, self).__init__()

        model = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0),
            nn.LeakyReLU(0.2, True),
            ResidualBlock(ndf),

            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            ResidualBlock(ndf * 2),

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
            ResidualBlock(ndf * 4),

            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True),
            ResidualBlock(ndf * 8),

            nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True),
            ResidualBlock(ndf * 8),

            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid(),
            nn.AdaptiveAvgPool2d(1)  # 新增一个自适应平均池化层,压缩为1x1
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input).view(-1)  # 将输出拉平为一维向量


def discriminator_loss(real_out, fake_out):
    d_loss = -1 * (torch.log(real_out) + torch.log(1 - fake_out))
    return d_loss.mean()


class VAEGANModelLoader(ModelLoader):
    def __init__(self, train_dataset, test_dataset, batch_size, model_path: str, discriminator_path: str, if_es=False,
                 if_debug=False):
        super().__init__(train_dataset, test_dataset, batch_size, model_path, if_es, if_debug)
        print('-' * 10, 'Loading VAE-GAN model', '-' * 10)
        # encoder
        self.latent_dim = latent_dim  # latent vector dimension
        encoder = VAEEncoder(self.latent_dim)
        # decoder
        decoder = VAEDecoder(self.latent_dim)
        # VAE
        self.model = VAEModel(encoder, decoder).to(self.device)
        self.lr = 1e-4  # learning rate
        # optimizer
        self.G_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.G_scheduler = CosineAnnealingLR(self.G_optimizer, T_max=20, eta_min=1e-5)

        # discriminator
        self.discriminator = Discriminator(input_channel).to(self.device)

        self.D_lr = 1e-8
        self.D_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.D_lr)
        self.D_scheduler = CosineAnnealingLR(self.D_optimizer, T_max=20, eta_min=0)

        self.criterion = nn.BCELoss()
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        # self.D_scheduler = CosineAnnealingLR(self.D_optimizer, T_max=50, eta_min=1e-5)

        self.discriminator_path = discriminator_path

        self.real_labels = torch.ones(batch_size).to(self.device)
        self.fake_labels = torch.zeros(batch_size).to(self.device)

        # load exist model
        self.load_model()

    def load_model(self):
        # load model weight
        model_dir = os.path.dirname(self.model_path)
        print('Try to load model from', self.model_path)
        # 检查模型文件夹路径是否存在
        if not os.path.exists(model_dir):
            # 不存在就创建新的目录
            os.makedirs(model_dir)
            print(f"Created directory '{model_dir}' for saving models.")
        if os.path.isfile(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                print("VAE model loaded successfully from '{}'".format(self.model_path))
            except Exception as e:
                print("Failed to load VAE model. Starting from scratch. Error: ", e)
        else:
            print("No saved model found at '{}'. Starting from scratch.".format(self.model_path))
        if os.path.isfile(self.discriminator_path):
            try:
                self.discriminator.load_state_dict(torch.load(self.discriminator_path, map_location=self.device))
                print("Discriminator A model loaded successfully from '{}'".format(self.discriminator_path))
            except Exception as e:
                print("Failed to load discriminator A model. Starting from scratch. Error: ", e)
        else:
            print("No saved model found at '{}'. Starting from scratch.".format(self.discriminator_path))

    def __test_epoch_vae(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, x in enumerate(self.test_iterator):
                # reshape the data
                x = x.view(-1, input_channel, 512, 512)
                x = x.to(self.device)

                # forward pass
                x_sample, z_mu, z_var = self.model(x)

                recon_loss = F.mse_loss(x_sample, x, reduction='sum')
                kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1.0 - z_var)

                g_loss = recon_loss * recon_weight + kl_loss * kl_weight

                test_loss += g_loss.item()

                print(f'VAE test batch {i}, g_loss: {g_loss.item() / self.batch_size}')

        return test_loss / len(self.test_dataset)

    def save_dis_model(self):
        """
            保存判别器的权重。
        """
        torch.save(self.discriminator.state_dict(), self.discriminator_path)
        print(f'Model saved to {self.discriminator_path}')

    def train(self, epochs=100, test_interval=10):
        train_losses_G = []
        train_losses_D = []
        test_losses_G = []

        for epoch in range(epochs):
            torch.cuda.empty_cache()
            train_loss_G = 0
            train_loss_D = 0

            for batch, x in enumerate(self.train_iterator):
                # 转换图像张量的形状并移至指定的设备
                x = x.view(-1, input_channel, 512, 512)
                x = x.to(self.device)

                # 对每张图片应用随机旋转
                rotation_angles = torch.randint(0, 4, (x.size(0),)) * 90
                for i in range(x.size(0)):
                    x[i] = rotate_image(x[i], rotation_angles[i].item())

                # 复制x到x_，以便在x_上添加噪声，同时保持x不变
                x_ = x.clone()

                for idx in range(x_.size(0)):
                    image = x_[idx]  # 获取单张图片

                    # image_ = image.clone()

                    random_judgement = random.random()
                    if random_judgement <= 0.25:
                        # 增加随机数量、随机大小的黑色小方块
                        # print('加入小方块')
                        image = grid_mask(image)
                    elif random_judgement <= 0.5:
                        # print('加入长条')
                        image = add_gray_stripes(image)
                    else:
                        # print('无修改')
                        pass

                    random_judgement = random.random()
                    if random_judgement <= 0.15:
                        # 加椒盐噪声
                        # print('椒盐噪声')
                        image = add_salt_pepper_noise(image)
                    elif random_judgement <= 0.3:
                        # 加高斯噪声
                        # print('高斯噪声')
                        image = add_gaussian_noise(image)
                    elif random_judgement <= 0.5:
                        # 在图像中心附近做一点扭曲
                        # print('加入扭曲')
                        image = add_central_distortion(image)
                    elif random_judgement <= 0.7:
                        # 随机向图像中添加一块模糊遮罩
                        # print('加入模糊')
                        image = random_kernel_filter(image)
                    elif random_judgement <= 0.9:
                        image = add_stripe_noise_pattern_around_image(image)
                    else:
                        # print('无噪声')
                        pass

                    x_[idx] = image

                    # 显示原始图片和带噪声的图片
                    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                    # ax1.imshow(image_.squeeze().cpu().numpy(), cmap="gray")
                    # ax1.set_title("Original Image")
                    # ax1.axis("off")
                    # ax2.imshow(image.squeeze().cpu().numpy(), cmap="gray")
                    # ax2.set_title("Noisy Image")
                    # ax2.axis("off")
                    # plt.tight_layout()
                    # plt.show()

                # 生成器训练
                self.G_optimizer.zero_grad()

                # 生成两张不同风格的图像y和z
                y, z_mu_y, z_var_y = self.model(x_)
                z, z_mu_z, z_var_z = self.model(y)

                # 计算重建损失和KL散度损失
                recon_loss_y = F.mse_loss(y, x, reduction='sum')
                kl_loss_y = 0.5 * torch.sum(torch.exp(z_var_y) + z_mu_y ** 2 - 1.0 - z_var_y)
                # 再次重建损失
                regen_loss = F.mse_loss(x, z, reduction='sum')
                kl_loss_z = 0.5 * torch.sum(torch.exp(z_var_z) + z_mu_z ** 2 - 1.0 - z_var_z)

                # 对生成的图像进行判别
                loss_G_adv = self.criterion_GAN(self.discriminator(y), self.real_labels) + \
                             self.criterion_GAN(self.discriminator(z), self.real_labels)

                # 计算生成器的总损失
                loss_G = recon_loss_y * recon_weight + (
                        kl_loss_y + kl_loss_z) * kl_weight + loss_G_adv * adv_weight + regen_loss * regen_weight
                loss_G.backward()
                self.G_optimizer.step()

                # 判别器训练
                self.D_optimizer.zero_grad()
                real_loss = self.criterion_GAN(self.discriminator(x), self.real_labels)
                fake_loss_y = self.criterion_GAN(self.discriminator(y.detach()), self.fake_labels)
                fake_loss_z = self.criterion_GAN(self.discriminator(z.detach()), self.fake_labels)
                loss_D = (real_loss + fake_loss_y + fake_loss_z) / 3
                loss_D.backward()
                self.D_optimizer.step()

                train_loss_G += loss_G.item()
                train_loss_D += loss_D.item()

                # 打印损失
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch + 1}/{len(self.train_iterator)}], "
                      f"D Loss: {loss_D.item():.4f}, G Loss: {loss_G.item() / self.batch_size:.4f}")

                self.G_scheduler.step()
                self.D_scheduler.step()

            # 计算平均训练损失
            train_loss_G /= len(self.train_dataset)
            train_loss_D /= len(self.train_dataset)

            train_losses_G.append(train_loss_G)
            train_losses_D.append(train_loss_D)

            test_loss_g = self.__test_epoch_vae()
            test_losses_G.append(test_loss_g)

            print(
                f'Epoch {epoch + 1}, Train G Loss: {train_loss_G:.6f}, Train D Loss: {train_loss_D:.6f}, '
                f'Test G Loss: {test_loss_g}')

            # 保存模型
            self.save_model()
            self.save_dis_model()

            # 按照间隔测试模型
            if (epoch + 1) % test_interval == 0:
                # 获取test_imgs目录下的所有图片文件
                test_imgs_dir = "./test_imgs/"
                test_imgs_files = [f for f in os.listdir(test_imgs_dir) if f.endswith(".bmp") or f.endswith(".png")]

                # 对test_imgs目录下的图片进行重建测试
                for _, file in enumerate(test_imgs_files):
                    file_path = os.path.join(test_imgs_dir, file)
                    output_name = f"epoch_{epoch + 1}_test_img_{_ + 1}"
                    self.regenerate_test(Image.open(file_path), output_name)

        # 将误差写入txt文件
        with open('cycle_gan_losses.txt', 'w') as f:
            f.write('Epoch\tTrain_G_Loss\tTrain_D_Loss\tTest_G_Loss\n')
            for batch in range(epochs):
                f.write(
                    f'{batch}\t{train_losses_G[batch]:.6f}\t{train_losses_D[batch]:.6f}\t{test_losses_G[batch]}\n')

    def regenerate_test(self, input_image: Image, file_name: str):
        """
        根据输入图片解码并重新生成测试模型效果
        :param input_image: 输入图片
        :param file_name: 输出图片名称
        :return: None
        """
        print('-' * 3, 'regenerate_test', '-' * 3)
        if not os.path.exists('./train_result/'):
            os.makedirs('./train_result/')
        # 定义预处理转换链
        if input_channel == 3:
            transform = transforms.Compose([
                transforms.Resize((512, 512)),  # 将图像调整为512x512
                transforms.Lambda(convert_to_rgb),  # 确保图像为三通道
                transforms.ToTensor()  # 将图像转换为PyTorch张量
            ])
        elif input_channel == 1:
            cv_image = np.array(input_image.convert('RGB'))
            # 转换为BGR格式
            cv_image = cv_image[:, :, ::-1]
            cv_image = img_pre_processing_gray(cv_image)  # 返回二值化后的图
            input_image = Image.fromarray(cv_image)
            transform = transforms.Compose([
                transforms.Resize((512, 512)),  # 将图像调整为512x512
                transforms.ToTensor()  # 将图像转换为PyTorch张量
            ])
        # 应用预处理转换链
        input_tensor = transform(input_image)

        # 添加批次维度并将图像输入模型
        input_tensor = input_tensor.to(self.device).unsqueeze(0)  # 添加批次维度，即从C x H x W变为1 x C x H x W

        with torch.no_grad():
            # 编码图像，获取潜在空间的均值和方差对数
            z_mu, z_log_var = self.model.encoder(input_tensor)

            # 从标准正态分布中采样epsilon
            # std = torch.exp(z_log_var / 2)
            # eps = torch.randn_like(std)
            # z = z_mu + eps * std

            # 解码
            regenerated_image = self.model.decoder(z_mu)

        regenerated_image = regenerated_image.cpu()
        # 图片名称：添加参数picture_name和索引i
        filename = f'./train_result/{file_name}.png'
        save_image(regenerated_image, filename, normalize=True)
        print('result saved to', filename)

    def test(self):
        # 获取test_imgs目录下的所有图片文件
        test_imgs_dir = "./test_imgs/"
        test_imgs_files = [f for f in os.listdir(test_imgs_dir) if f.endswith(".bmp") or f.endswith(".png")]

        # 对test_imgs目录下的图片进行重建测试
        for i, file in enumerate(test_imgs_files):
            file_path = os.path.join(test_imgs_dir, file)
            output_name = f"test_result_{i + 1}"
            self.regenerate_test(Image.open(file_path), output_name)

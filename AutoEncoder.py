import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from ImagePreProcessing import *
from ImageTools import *
from ModelLoader import ModelLoader

latent_dim = 256
input_channel = 1
# max_size = 1536
max_size = 1792


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # Squeeze operation
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation operation
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y_fc = self.fc(y).view(b, c, 1, 1)
        y_ex = y_fc.expand(x.size())
        return x * y_ex


class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes):
        super(SpatialPyramidPooling, self).__init__()
        self.pool_sizes = pool_sizes

    def forward(self, x):
        features = []
        for pool_size in self.pool_sizes:
            features.append(F.adaptive_avg_pool2d(x, pool_size).view(x.size(0), -1))
        return torch.cat(features, 1)


class VAEEncoder(nn.Module):
    def __init__(self, z_dim):
        super(VAEEncoder, self).__init__()

        # Initial batch normalization
        self.initial_norm = nn.InstanceNorm2d(input_channel)

        # Encoder architecture: Five sets of [BN -> Conv2d] for RGB images
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, max_size // 16, kernel_size=4, stride=2, padding=1),
            # Output: [max_size // 16, 256, 256]
            nn.InstanceNorm2d(max_size // 16),
            SELayer(max_size // 16),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(max_size // 16, max_size // 8, kernel_size=4, stride=2, padding=1),
            # Output: [max_size // 8, 128, 128]
            nn.InstanceNorm2d(max_size // 8),
            SELayer(max_size // 8),
            nn.LeakyReLU(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(max_size // 8, max_size // 4, kernel_size=4, stride=2, padding=1),
            # Output: [max_size // 4, 64, 64]
            nn.InstanceNorm2d(max_size // 4),
            SELayer(max_size // 4),
            nn.LeakyReLU(0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(max_size // 4, max_size // 2, kernel_size=4, stride=2, padding=1),
            # Output: [max_size // 2, 32, 32]
            nn.InstanceNorm2d(max_size // 2),
            SELayer(max_size // 2),
            nn.LeakyReLU(0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(max_size // 2, max_size, kernel_size=4, stride=2, padding=1),
            # Output: [max_size, 16, 16]
            nn.InstanceNorm2d(max_size),
            SELayer(max_size),
            nn.LeakyReLU(0.2),
        )

        # Residual connections
        self.skip1 = nn.Conv2d(input_channel, max_size // 16, kernel_size=1, stride=2, padding=0)
        self.skip2 = nn.Conv2d(max_size // 16, max_size // 8, kernel_size=1, stride=2, padding=0)
        self.skip3 = nn.Conv2d(max_size // 8, max_size // 4, kernel_size=1, stride=2, padding=0)
        self.skip4 = nn.Conv2d(max_size // 4, max_size // 2, kernel_size=1, stride=2, padding=0)
        self.skip5 = nn.Conv2d(max_size // 2, max_size, kernel_size=1, stride=2, padding=0)

        # Add SPP layer
        self.spp = SpatialPyramidPooling([1, 2, 4])
        # Calculate the flattened size after the SPP layer
        spp_total_size = max_size * (1 * 1 + 2 * 2 + 4 * 4)
        self.fc_mu = nn.Linear(spp_total_size, z_dim)
        self.fc_var = nn.Linear(spp_total_size, z_dim)

        # 初始化
        init.kaiming_uniform_(self.conv1[0].weight, a=0.2, nonlinearity='leaky_relu')
        init.kaiming_uniform_(self.conv2[0].weight, a=0.2, nonlinearity='leaky_relu')
        init.kaiming_uniform_(self.conv3[0].weight, a=0.2, nonlinearity='leaky_relu')
        init.kaiming_uniform_(self.conv4[0].weight, a=0.2, nonlinearity='leaky_relu')
        init.kaiming_uniform_(self.conv5[0].weight, a=0.2, nonlinearity='leaky_relu')
        init.xavier_uniform_(self.fc_mu.weight)
        self.fc_mu.bias.data.fill_(0)
        init.xavier_uniform_(self.fc_var.weight)
        self.fc_var.bias.data.fill_(0)

    def forward(self, x):
        # Apply initial batch normalization
        x_ini = self.initial_norm(x)

        # Apply the five sets of [Conv2d -> BN -> ReLU] with residual connections
        identity1 = self.skip1(x_ini)
        x1 = self.conv1(x_ini) + identity1

        identity2 = self.skip2(x1)
        x2 = self.conv2(x1) + identity2

        identity3 = self.skip3(x2)
        x3 = self.conv3(x2) + identity3

        identity4 = self.skip4(x3)
        x4 = self.conv4(x3) + identity4

        identity5 = self.skip5(x4)
        x5 = self.conv5(x4) + identity5

        # Pass through the SPP layer
        x_spp = self.spp(x5)

        # Flatten the output for the fully connected layers
        x_final = x_spp.view(x_spp.size(0), -1)

        # Pass through the fully connected layers
        z_mu = self.fc_mu(x_final)
        z_var = self.fc_var(x_final)

        return z_mu, z_var


class VAEDecoder(nn.Module):
    def __init__(self, z_dim):
        super(VAEDecoder, self).__init__()

        self.feature_map_size = max_size * 16 * 16  # [max_size, 16, 16]

        self.fc = nn.Linear(z_dim, self.feature_map_size)

        self.conv_transpose1 = nn.Sequential(
            nn.ConvTranspose2d(max_size, max_size // 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(max_size // 2),
            SELayer(max_size // 2),
            nn.LeakyReLU(0.2),
        )
        self.conv_transpose2 = nn.Sequential(
            nn.ConvTranspose2d(max_size // 2, max_size // 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(max_size // 4),
            SELayer(max_size // 4),
            nn.LeakyReLU(0.2),
        )
        self.conv_transpose3 = nn.Sequential(
            nn.ConvTranspose2d(max_size // 4, max_size // 8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(max_size // 8),
            SELayer(max_size // 8),
            nn.LeakyReLU(0.2),
        )
        self.conv_transpose4 = nn.Sequential(
            nn.ConvTranspose2d(max_size // 8, max_size // 16, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(max_size // 16),
            SELayer(max_size // 16),
            nn.LeakyReLU(0.2),
        )
        self.conv_transpose5 = nn.Sequential(
            nn.ConvTranspose2d(max_size // 16, input_channel, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # 使用Tanh激活函数将像素值限制在-1到1之间
        )

        # 残差连接
        self.skip1 = nn.ConvTranspose2d(max_size, max_size // 2, kernel_size=4, stride=2, padding=1)
        self.skip2 = nn.ConvTranspose2d(max_size // 2, max_size // 4, kernel_size=4, stride=2, padding=1)
        self.skip3 = nn.ConvTranspose2d(max_size // 4, max_size // 8, kernel_size=4, stride=2, padding=1)
        self.skip4 = nn.ConvTranspose2d(max_size // 8, max_size // 16, kernel_size=4, stride=2, padding=1)
        self.skip5 = nn.ConvTranspose2d(max_size // 16, input_channel, kernel_size=4, stride=2, padding=1)

        self._initialize_weights()

    def forward(self, x):
        x_fc = self.fc(x)
        x_viewed = x_fc.view(-1, max_size, 16, 16)

        identity1 = self.skip1(x_viewed)
        x1 = self.conv_transpose1(x_viewed) + identity1

        identity2 = self.skip2(x1)
        x2 = self.conv_transpose2(x1) + identity2

        identity3 = self.skip3(x2)
        x3 = self.conv_transpose3(x2) + identity3

        identity4 = self.skip4(x3)
        x4 = self.conv_transpose4(x3) + identity4

        identity5 = self.skip5(x4)
        x5 = self.conv_transpose5(x4) + identity5

        return x5

    def _initialize_weights(self):
        init.kaiming_uniform_(self.conv_transpose1[0].weight, a=0.2, nonlinearity='leaky_relu')
        init.kaiming_uniform_(self.conv_transpose2[0].weight, a=0.2, nonlinearity='leaky_relu')
        init.kaiming_uniform_(self.conv_transpose3[0].weight, a=0.2, nonlinearity='leaky_relu')
        init.kaiming_uniform_(self.conv_transpose4[0].weight, a=0.2, nonlinearity='leaky_relu')
        init.xavier_uniform_(self.conv_transpose5[0].weight)
        self.conv_transpose5[0].bias.data.fill_(0)
        init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)


class VAEModel(nn.Module):
    def __init__(self, encoder: VAEEncoder, decoder: VAEDecoder):
        super(VAEModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        # encode
        z_mu, z_var = self.encoder(x)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std) + z_mu

        # decode
        predicted = self.decoder(x_sample)
        return predicted, z_mu, z_var


class VAEModelLoader(ModelLoader):
    def __init__(self, train_dataset, test_dataset, batch_size, model_path: str, if_early_stop=False, debug_mode=False):
        super().__init__(train_dataset, test_dataset, batch_size, model_path, if_early_stop, debug_mode)
        print('-' * 10, 'Loading VAE model', '-' * 10)
        # encoder
        self.latent_dim = latent_dim  # latent vector dimension
        encoder = VAEEncoder(self.latent_dim)
        # decoder
        decoder = VAEDecoder(self.latent_dim)
        # VAE
        self.model = VAEModel(encoder, decoder).to(self.device)
        self.lr = 1e-4  # learning rate
        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # 设置学习率调度
        # 余弦调度
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-5)

        # load exist model
        self.load_model()

        self.train_losses = []
        self.test_losses = []

    def _train_epoch(self):
        # set the train mode
        self.model.train()
        # loss of the epoch
        train_loss = 0
        for i, x in enumerate(self.train_iterator):
            # reshape the data into [batch_size, 3, 512, 512]
            x = x.view(-1, input_channel, 512, 512)  # 后面需要conv，所以先调整size
            x = x.to(self.device)

            self.optimizer.zero_grad()

            x_sample, z_mu, z_var = self.model(x)

            # reconstruction loss
            # recon_loss = F.binary_cross_entropy(x_sample, x, reduction='sum')
            recon_loss = F.mse_loss(x_sample, x, reduction='sum')

            # kl divergence loss
            kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1.0 - z_var)

            # total loss
            loss = recon_loss + kl_loss

            loss.backward()
            train_loss += loss.item()

            self.optimizer.step()

            print(f'Train batch {i}, loss: {loss.item() / self.batch_size}')

        return train_loss / len(self.train_dataset)

    def _test_epoch(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, x in enumerate(self.test_iterator):
                # reshape the data
                x = x.view(-1, input_channel, 512, 512)
                x = x.to(self.device)

                # forward pass
                x_sample, z_mu, z_var = self.model(x)

                # reconstruction loss
                # recon_loss = F.binary_cross_entropy(x_sample, x, reduction='sum')
                recon_loss = F.mse_loss(x_sample, x, reduction='sum')

                # kl divergence loss
                kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1.0 - z_var)

                # total loss
                loss = recon_loss + kl_loss
                test_loss += loss.item()

                print(f'Test batch {i}, loss: {loss.item() / self.batch_size}')

        return test_loss / len(self.test_dataset)

    def train(self, epochs=50, test_interval=10):
        if self.if_early_stop:
            # 早停策略防止过拟合
            best_test_loss = float('inf')
            patience_counter = 0

        for e in range(epochs):
            print('-' * 10, 'Train epoch', e, 'started!', '-' * 10)
            train_loss = self._train_epoch()
            test_loss = self._test_epoch()
            print(f'Epoch {e}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}')

            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)

            if self.scheduler:
                self.scheduler.step()
            # 保存模型
            self.save_model(test_loss)
            # 按照间隔测试模型
            if (e + 1) % test_interval == 0:
                # 获取test_imgs目录下的所有图片文件
                test_imgs_dir = "./test_imgs/"
                test_imgs_files = [f for f in os.listdir(test_imgs_dir) if f.endswith(".bmp")]

                # 对test_imgs目录下的图片进行重建测试
                for i, file in enumerate(test_imgs_files):
                    file_path = os.path.join(test_imgs_dir, file)
                    output_name = f"epoch_{e + 1}_test_img_{i + 1}"
                    self.regenerate_test(Image.open(file_path), output_name)

            if self.if_early_stop:
                # 计算早停累计
                if best_test_loss > test_loss:
                    best_test_loss = test_loss
                    patience_counter = 1
                else:
                    patience_counter += 1
                if patience_counter > max(epochs / 5, 10):
                    # 早停
                    print('Training interrupted to avoid overfitting.')
                    break

        print(f'Final Train Loss: {self.train_losses[-1]:.2f}, Final Test Loss: {self.test_losses[-1]:.2f}')

        # 将 train_losses 和 test_losses 保存到文件中
        with open('losses.txt', 'w') as f:
            f.write('Train Losses:\n')
            f.write(', '.join(map(str, self.train_losses)))
            f.write('\n\nTest Losses:\n')
            f.write(', '.join(map(str, self.test_losses)))

    def random_generate_test(self, test_time=10, picture_name='test_image'):
        """
        根据特征向量分布随机生成特征向量并解码出图片
        :param test_time: 测试次数
        :param picture_name: 保存图片名称
        :return: None
        """
        print('-' * 3, 'random_generate_test', '-' * 3)
        if not os.path.exists('./VAE_test/'):
            os.makedirs('./VAE_test/')

        self.model.eval()
        with torch.no_grad():
            for i in range(test_time):
                z = torch.randn(1, self.latent_dim).to(self.device)
                reconstructed_img = self.model.decoder(z)
                img = reconstructed_img.cpu().squeeze(0)  # 从batch中移除，得到3x512x512的图片
                img = img.permute(1, 2, 0)  # 调整为512x512x3

                # 将张量数据转换为PIL图像
                img = (img.numpy() * 255).astype(np.uint8)
                img = Image.fromarray(img)

                filename = f'./VAE_test/{picture_name}_{i}.png'
                img.save(filename)
                print('result saved to', filename)

    def regenerate_test(self, input_image: Image, file_name: str):
        """
        根据输入图片解码并重新生成测试模型效果
        :param input_image: 输入图片
        :param file_name: 输出图片名称
        :return: None
        """
        print('-' * 3, 'regenerate_test', '-' * 3)
        if not os.path.exists('./VAE_test/'):
            os.makedirs('./VAE_test/')
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

        img = regenerated_image.cpu().squeeze(0)  # 从batch中移除，得到inputchannel x 512 x 512的图片
        # print(img.shape)  # torch.Size([1, 512, 512])
        img = img.permute(1, 2, 0)  # 调整为512x512x input_channel
        # print(img.shape)  # torch.Size([512, 512, 1])

        if img.shape[2] == 1:
            img = img.squeeze()  # 去除单一通道维度
            img_np = img.numpy()
            if img_np.dtype == np.float32 or img_np.dtype == np.float64:
                img_np = (img_np * 255).astype(np.uint8)
            img = Image.fromarray(img_np, mode='L')  # 'L' 模式代表灰度图
        elif img.shape[2] == 3:
            img_np = (img.numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_np)

        # 图片名称：添加参数picture_name和索引i
        filename = f'./VAE_test/{file_name}.png'
        img.save(filename)  # 直接保存图片
        print('result saved to', filename)

    def test(self):
        # 获取test_imgs目录下的所有图片文件
        test_imgs_dir = "./test_imgs/"
        test_imgs_files = [f for f in os.listdir(test_imgs_dir) if f.endswith(".bmp")]

        # 对test_imgs目录下的图片进行重建测试
        for i, file in enumerate(test_imgs_files):
            file_path = os.path.join(test_imgs_dir, file)
            output_name = f"test_result_{i + 1}"
            self.regenerate_test(Image.open(file_path), output_name)

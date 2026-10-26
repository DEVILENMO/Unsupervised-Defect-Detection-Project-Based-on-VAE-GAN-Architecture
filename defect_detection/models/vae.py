"""VAE 生成器: 编码器 + 解码器。

编码器: 5 组 [Conv -> InstanceNorm -> SE -> LeakyReLU] + 1x1 卷积残差跳连, 末端接空间金字塔池化 (SPP)。
解码器: 结构对称的 5 组转置卷积, 输出经 Tanh 归一到 [-1, 1]。

注意: 各子层的属性名参与 state_dict 键名, 与历史 checkpoint 对应, 不要重命名。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from defect_detection.config import BASE_CHANNEL, INPUT_CHANNEL, LATENT_DIM


class SELayer(nn.Module):
    """Squeeze-and-Excitation 通道注意力层。"""

    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y_fc = self.fc(y).view(b, c, 1, 1)
        y_ex = y_fc.expand(x.size())
        return x * y_ex


class SpatialPyramidPooling(nn.Module):
    """空间金字塔池化, 把任意尺寸特征图压成固定长度向量。"""

    def __init__(self, pool_sizes):
        super().__init__()
        self.pool_sizes = pool_sizes

    def forward(self, x):
        features = []
        for pool_size in self.pool_sizes:
            features.append(F.adaptive_avg_pool2d(x, pool_size).view(x.size(0), -1))
        return torch.cat(features, 1)


class VAEEncoder(nn.Module):
    def __init__(self, z_dim: int = LATENT_DIM):
        super().__init__()
        ch = BASE_CHANNEL

        self.initial_norm = nn.InstanceNorm2d(INPUT_CHANNEL)

        # 5 组 [Conv -> IN -> SE -> LeakyReLU], 输入 512x512 时分辨率依次为 256/128/64/32/16
        self.conv1 = nn.Sequential(
            nn.Conv2d(INPUT_CHANNEL, ch // 16, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ch // 16),
            SELayer(ch // 16),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch // 16, ch // 8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ch // 8),
            SELayer(ch // 8),
            nn.LeakyReLU(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch // 8, ch // 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ch // 4),
            SELayer(ch // 4),
            nn.LeakyReLU(0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(ch // 4, ch // 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ch // 2),
            SELayer(ch // 2),
            nn.LeakyReLU(0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(ch // 2, ch, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ch),
            SELayer(ch),
            nn.LeakyReLU(0.2),
        )

        # 1x1 卷积残差跳连
        self.skip1 = nn.Conv2d(INPUT_CHANNEL, ch // 16, kernel_size=1, stride=2)
        self.skip2 = nn.Conv2d(ch // 16, ch // 8, kernel_size=1, stride=2)
        self.skip3 = nn.Conv2d(ch // 8, ch // 4, kernel_size=1, stride=2)
        self.skip4 = nn.Conv2d(ch // 4, ch // 2, kernel_size=1, stride=2)
        self.skip5 = nn.Conv2d(ch // 2, ch, kernel_size=1, stride=2)

        # SPP 输出展平后接全连接, 映射到潜空间的均值与对数方差
        self.spp = SpatialPyramidPooling([1, 2, 4])
        spp_total_size = ch * (1 * 1 + 2 * 2 + 4 * 4)
        self.fc_mu = nn.Linear(spp_total_size, z_dim)
        self.fc_var = nn.Linear(spp_total_size, z_dim)

        for conv in (self.conv1, self.conv2, self.conv3, self.conv4, self.conv5):
            init.kaiming_uniform_(conv[0].weight, a=0.2, nonlinearity="leaky_relu")
        init.xavier_uniform_(self.fc_mu.weight)
        self.fc_mu.bias.data.fill_(0)
        init.xavier_uniform_(self.fc_var.weight)
        self.fc_var.bias.data.fill_(0)

    def forward(self, x):
        x_ini = self.initial_norm(x)

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

        x_spp = self.spp(x5)
        x_final = x_spp.view(x_spp.size(0), -1)

        return self.fc_mu(x_final), self.fc_var(x_final)


class VAEDecoder(nn.Module):
    def __init__(self, z_dim: int = LATENT_DIM):
        super().__init__()
        ch = BASE_CHANNEL

        # [ch, 16, 16]
        self.feature_map_size = ch * 16 * 16
        self.fc = nn.Linear(z_dim, self.feature_map_size)

        self.conv_transpose1 = nn.Sequential(
            nn.ConvTranspose2d(ch, ch // 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ch // 2),
            SELayer(ch // 2),
            nn.LeakyReLU(0.2),
        )
        self.conv_transpose2 = nn.Sequential(
            nn.ConvTranspose2d(ch // 2, ch // 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ch // 4),
            SELayer(ch // 4),
            nn.LeakyReLU(0.2),
        )
        self.conv_transpose3 = nn.Sequential(
            nn.ConvTranspose2d(ch // 4, ch // 8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ch // 8),
            SELayer(ch // 8),
            nn.LeakyReLU(0.2),
        )
        self.conv_transpose4 = nn.Sequential(
            nn.ConvTranspose2d(ch // 8, ch // 16, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ch // 16),
            SELayer(ch // 16),
            nn.LeakyReLU(0.2),
        )
        self.conv_transpose5 = nn.Sequential(
            nn.ConvTranspose2d(ch // 16, INPUT_CHANNEL, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        # 残差跳连
        self.skip1 = nn.ConvTranspose2d(ch, ch // 2, kernel_size=4, stride=2, padding=1)
        self.skip2 = nn.ConvTranspose2d(ch // 2, ch // 4, kernel_size=4, stride=2, padding=1)
        self.skip3 = nn.ConvTranspose2d(ch // 4, ch // 8, kernel_size=4, stride=2, padding=1)
        self.skip4 = nn.ConvTranspose2d(ch // 8, ch // 16, kernel_size=4, stride=2, padding=1)
        self.skip5 = nn.ConvTranspose2d(ch // 16, INPUT_CHANNEL, kernel_size=4, stride=2, padding=1)

        self._initialize_weights()

    def forward(self, x):
        x_fc = self.fc(x)
        x_viewed = x_fc.view(-1, BASE_CHANNEL, 16, 16)

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
        for layer in (self.conv_transpose1, self.conv_transpose2, self.conv_transpose3,
                      self.conv_transpose4):
            init.kaiming_uniform_(layer[0].weight, a=0.2, nonlinearity="leaky_relu")
        init.xavier_uniform_(self.conv_transpose5[0].weight)
        self.conv_transpose5[0].bias.data.fill_(0)
        init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)


class VAEModel(nn.Module):
    """编码 + 重参数化采样 + 解码。"""

    def __init__(self, encoder: VAEEncoder, decoder: VAEDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z_mu, z_var = self.encoder(x)

        # 重参数化: std = exp(log_var / 2)
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std) + z_mu

        predicted = self.decoder(x_sample)
        return predicted, z_mu, z_var


def build_vae(checkpoint_path=None, device="cpu") -> VAEModel:
    """构建 VAE 并可选加载权重, 供推理与训练入口复用。"""
    model = VAEModel(VAEEncoder(LATENT_DIM), VAEDecoder(LATENT_DIM)).to(device)
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"VAE weights loaded from '{checkpoint_path}'")
    return model

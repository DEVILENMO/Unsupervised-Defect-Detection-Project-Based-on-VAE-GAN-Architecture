"""PatchGAN 风格判别器 (来自 CycleGAN), 末端自适应池化为单个概率值。

用途:
1. VAE-GAN 训练时提供局部判别反馈;
2. 推理时判断 SAM 切割结果是否为真实目标 (见 target_discriminator 模块)。
"""

import torch
import torch.nn as nn

from defect_detection.config import INPUT_CHANNEL


class ResidualBlock(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc: int = INPUT_CHANNEL, ndf: int = 64):
        super().__init__()
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
            # 自适应平均池化把 patch 概率图压成单个概率
            nn.AdaptiveAvgPool2d(1),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input).view(-1)

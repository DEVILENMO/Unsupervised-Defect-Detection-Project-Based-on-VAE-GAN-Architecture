"""基于 VAE-GAN 重建的无监督缺陷检测核心包。

子模块:
- config: 路径与常量
- models: 网络结构 (VAE / 判别器)
- data: 数据集加载与增强
- segmentation: SAM 目标切割
- target_discriminator: 切割结果判别
- training: 训练器
- meta_learning: MAML 元学习
- detector: 缺陷检测推理管线
- image_utils: 图像读写与预处理工具
"""

__version__ = "1.1.0"

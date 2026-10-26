"""训练器: 通用训练循环骨架与各模型的具体实现。"""

from defect_detection.training.base import ModelLoader
from defect_detection.training.vae_trainer import VAEModelLoader
from defect_detection.training.vae_gan_trainer import VAEGANModelLoader

__all__ = ["ModelLoader", "VAEModelLoader", "VAEGANModelLoader"]

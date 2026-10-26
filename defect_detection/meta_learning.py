"""MAML 元学习: 用通用样本 + 特定任务样本训练泛化初值 (冷启动) 模型, 避免过拟合。"""

import os
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from defect_detection.config import CUT_IMAGE_DIR, LATENT_DIM, TEST_DATASET_FILE
from defect_detection.data.augment import do_nothing
from defect_detection.data.dataset import TensorImageDataset, augment_tensor_dataset, load_image_datasets
from defect_detection.models.vae import VAEDecoder, VAEEncoder, VAEModel


class MAML:
    """Model-Agnostic Meta-Learning 包装的 VAE。

    inner_lr: 任务内快速适配的学习率; beta: 元更新步长。
    """

    def __init__(self, inner_lr: float, beta: float, device):
        self.device = device
        encoder = VAEEncoder(LATENT_DIM)
        decoder = VAEDecoder(LATENT_DIM)
        self.model = VAEModel(encoder, decoder).to(self.device)
        self.inner_lr = inner_lr
        self.beta = beta
        self.grad_clip_norm = 1.0

    def inner_update(self, x):
        """单个任务内的快速适配 (一步梯度更新)。"""
        x_sample, z_mu, z_var = self.model(x)
        inner_loss = compute_task_loss(x_sample, x, z_mu, z_var)
        self.model.zero_grad()
        inner_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
        inner_optimizer = optim.Adam(self.model.parameters(), lr=self.inner_lr)
        inner_optimizer.step()
        return inner_loss

    def meta_update(self, num_tasks, general_loader, specific_loader, num_samples_per_task):
        """一次元更新: 对若干任务的适配后梯度做平均, 更新初始参数。"""
        for _, param in self.model.named_parameters():
            if param.data.dtype != torch.float32:
                param.data = param.data.float()

        param_dict = deepcopy(self.model.state_dict())
        param_dict = {name: torch.zeros_like(p, dtype=torch.float32, requires_grad=True)
                      for name, p in param_dict.items()}

        for _ in range(num_tasks):
            x_task = sample_task_data(general_loader, specific_loader, num_samples_per_task,
                                      self.device)
            x_task_viewed = x_task.view(-1, 1, 512, 512)

            self.inner_update(x_task_viewed)
            updated_param = deepcopy(self.model.state_dict())

            x_query = sample_task_data(general_loader, specific_loader, num_samples_per_task,
                                       self.device)
            x_query_viewed = x_query.view(-1, 1, 512, 512)

            self.model.load_state_dict(updated_param)
            x_sample, z_mu, z_var = self.model(x_query_viewed)
            task_loss = compute_task_loss(x_sample, x_task_viewed, z_mu, z_var)
            print("\r\rtask_loss:", task_loss.item())
            self.model.zero_grad()
            task_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)

            # 以各参数梯度的均值作为该任务的元梯度 (原实现的近似方式)
            meta_grad = {}
            for name, params in zip(self.model.state_dict(), self.model.parameters()):
                if params.grad is not None:
                    meta_grad[name] = torch.mean(params.grad.data)

            for name in param_dict:
                if name in meta_grad:
                    param_dict[name] = param_dict[name] + meta_grad[name] * torch.ones_like(
                        param_dict[name])

        net_params = self.model.state_dict()
        net_params_new = {name: net_params[name] + self.beta * param_dict[name] / num_tasks
                          for name in net_params}
        self.model.load_state_dict(net_params_new)


def train_maml_vae(maml_model: MAML, general_loader, specific_loader, num_tasks,
                   num_inner_steps, num_samples_per_task, meta_iteration):
    """一轮元迭代: 执行若干次元更新。"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Meta Iteration: {meta_iteration}")

    for inner_step in range(num_inner_steps):
        maml_model.meta_update(num_tasks, general_loader, specific_loader, num_samples_per_task)
        print(f"\rInner Step: {inner_step + 1}/{num_inner_steps}")


def compute_task_loss(x_recon, x, z_mu, z_var):
    """任务损失 = 重建损失 + KL 散度。"""
    recon_loss = F.mse_loss(x_recon, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + z_var - z_mu.pow(2) - z_var.exp())
    return recon_loss + kl_loss


def sample_task_data(general_loader, specific_loader, num_samples, device):
    """从通用数据与特定任务数据中各采样 num_samples 张图, 拼成一个任务的批次。"""
    general_data = next(iter(general_loader))
    specific_data = next(iter(specific_loader))

    general_indices = torch.randint(0, len(general_data), (num_samples,))
    specific_indices = torch.randint(0, len(specific_data), (num_samples,))

    task_data = torch.cat((general_data[general_indices], specific_data[specific_indices]), dim=0)
    return task_data.to(device)


def seg_tensor_dataset(tensor_dataset):
    """把数据集前 5 张划为 part1, 其余划为 part2。"""
    part1, part2 = [], []
    for index, tensor in enumerate(tensor_dataset):
        if index < 5:
            part1.append(tensor)
        else:
            part2.append(tensor)
    return torch.stack(part1), torch.stack(part2)


def merge_datasets(tensor_dataset_list: list):
    """合并多个张量集合。"""
    merged = []
    for dataset in tensor_dataset_list:
        merged.extend(dataset)
    return torch.stack(merged)


def load_datasets(general_image_dir=CUT_IMAGE_DIR, test_dataset_file=TEST_DATASET_FILE):
    """加载元学习数据: 通用集 (SAM 切割图, 不做增强) 与特定任务集 (test_dataset.pt)。

    Returns:
        (general_data, specific_data); specific_data 在缓存文件缺失时为 None。
    """
    general_data = augment_tensor_dataset(load_image_datasets(general_image_dir),
                                          augment_fn=do_nothing)

    if os.path.isfile(test_dataset_file):
        specific_data = torch.load(test_dataset_file, weights_only=True)
        print(f"Test dataset has been loaded and merged from {test_dataset_file}.")
    else:
        print("Test dataset files not found.")
        specific_data = None
    return general_data, specific_data


def build_meta_loaders(general_data, specific_data, batch_size: int = 5):
    """构建元学习用的两个 DataLoader。"""
    general_loader = DataLoader(TensorImageDataset(general_data), batch_size=batch_size,
                                shuffle=True)
    specific_loader = DataLoader(TensorImageDataset(specific_data), batch_size=batch_size,
                                 shuffle=True)
    return general_loader, specific_loader

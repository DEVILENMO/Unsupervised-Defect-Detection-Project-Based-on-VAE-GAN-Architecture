from copy import deepcopy

from torch.utils.data import DataLoader

from AutoEncoder import *
from DataAugment import limit_size, do_nothing
from Datasets import *
from VAE_GAN_train import load_image_datasets


def augment_tensor_dataset(tensor_dataset):
    augmented_tensors = []
    for tensor in tensor_dataset:
        # augmented_images = augment_image_tensor(tensor)
        augmented_images = do_nothing(tensor)
        for i in range(len(augmented_images)):
            augmented_images[i] = limit_size(augmented_images[i])
        augmented_tensors.extend(augmented_images)
    return torch.stack(augmented_tensors)  # 将列表转换回一个新的张量


class MLMA:
    def __init__(self, inner_lr, beta, d):
        self.device = d
        encoder = VAEEncoder(latent_dim)
        decoder = VAEDecoder(latent_dim)
        self.model = VAEModel(encoder, decoder).to(self.device)
        self.inner_lr = inner_lr
        self.beta = beta
        self.grad_clip_norm = 1.0  # 添加梯度裁剪的范数值

    def inner_update(self, x):
        x_sample, z_mu, z_var = self.model(x)
        inner_loss = compute_task_loss(x_sample, x, z_mu, z_var)
        self.model.zero_grad()
        inner_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)  # 梯度裁剪
        inner_optimizer = optim.Adam(self.model.parameters(), lr=self.inner_lr)
        inner_optimizer.step()
        return inner_loss

    def meta_update(self, num_tasks, general_loader, specific_loader, num_samples_per_task):
        # 将模型参数转换为浮点类型
        for name, param in self.model.named_parameters():
            if param.data.dtype != torch.float32:
                param.data = param.data.float()

        param_dict = deepcopy(self.model.state_dict())
        param_dict = {name: torch.zeros_like(param_dict[name], dtype=torch.float32, requires_grad=True) for name in
                      param_dict}

        for _ in range(num_tasks):
            x_task = sample_task_data(general_loader, specific_loader, num_samples_per_task)
            x_task_viewed = x_task.view(-1, input_channel, 512, 512)

            self.inner_update(x_task_viewed)
            updated_param = deepcopy(self.model.state_dict())

            x_query = sample_task_data(general_loader, specific_loader, num_samples_per_task)
            x_query_viewed = x_query.view(-1, input_channel, 512, 512)

            self.model.load_state_dict(updated_param)
            x_sample, z_mu, z_var = self.model(x_query_viewed)
            task_loss = compute_task_loss(x_sample, x_task_viewed, z_mu, z_var)
            print('\r\rtast_loss:', task_loss.item())
            self.model.zero_grad()
            task_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)  # 梯度裁剪

            meta_grad = {}
            for name, params in zip(self.model.state_dict(), self.model.parameters()):
                if params.grad is not None:
                    meta_grad[name] = torch.mean(params.grad.data)  # 对梯度进行平均

            for name in param_dict:
                if name in meta_grad:
                    param_dict[name] = param_dict[name] + meta_grad[name] * torch.ones_like(
                        param_dict[name])  # 标量梯度乘以全1张量并累加

        net_params = self.model.state_dict()
        net_params_new = {name: net_params[name] + self.beta * param_dict[name] / num_tasks for name in net_params}
        self.model.load_state_dict(net_params_new)


def train_maml_vae(mlma_model, general_loader, specific_loader, num_tasks, num_inner_steps, num_samples_per_task,
                   meta_iteration):
    global device
    torch.cuda.empty_cache()
    print(f"Meta Iteration: {meta_iteration}")

    for inner_step in range(num_inner_steps):
        mlma_model.meta_update(num_tasks, general_loader, specific_loader, num_samples_per_task)
        print(f"\rInner Step: {inner_step + 1}/{num_inner_steps}")


def compute_task_loss(x_recon, x, z_mu, z_var):
    # 重建损失
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')

    # KL散度损失
    kl_loss = -0.5 * torch.sum(1 + z_var - z_mu.pow(2) - z_var.exp())

    # 总损失
    task_loss = recon_loss + kl_loss
    return task_loss


def sample_task_data(general_loader, specific_loader, num_samples):
    """
    Sample task-specific data from the general and specific data loaders.
    """
    general_data = next(iter(general_loader))
    specific_data = next(iter(specific_loader))

    general_indices = torch.randint(0, len(general_data), (num_samples,))
    specific_indices = torch.randint(0, len(specific_data), (num_samples,))

    task_data = torch.cat((general_data[general_indices], specific_data[specific_indices]), dim=0)
    global device
    return task_data.to(device)


def augment_image_tensor(image_tensor, zoom_factor=1.2):
    # 转换为PIL图像
    image = transforms.ToPILImage()(image_tensor)

    # # 获取图像尺寸
    # width, height = image.size
    #
    # # 计算裁剪区域的坐标
    # crop_size = int(min(width, height) / zoom_factor)
    # left = (width - crop_size) // 2
    # top = (height - crop_size) // 2
    # right = left + crop_size
    # bottom = top + crop_size
    #
    # # 裁剪图像
    # cropped_image = image.crop((left, top, right, bottom))

    # 将裁剪后的图像缩放回原始尺寸
    resized_image = image.resize((512, 512), resample=Image.BICUBIC)

    # 转换回张量
    resized_tensor = transforms.ToTensor()(resized_image)

    return resized_tensor


def seg_tensor_dataset(tensor_dataset):
    part1, part2 = [], []
    for index, tensor in enumerate(tensor_dataset):
        image = transforms.ToPILImage()(tensor)
        if index < 5:
            part1.append(transforms.ToTensor()(image))
        else:
            part2.append(transforms.ToTensor()(image))
    return torch.stack(part1), torch.stack(part2)


def merge_datasets(tensor_dataset_list: list):
    merged_dataset = []
    for dataset in tensor_dataset_list:
        merged_dataset.extend(dataset)
    return torch.stack(merged_dataset)


def load_datasets(zoom_factor=1.2):
    # 加载训练数据集
    # loaded_datasets = []
    # for i in range(1, 6):
    #     dataset_path = os.path.join(datasets_dir, f'train_dataset_{i}.pt')
    #     if os.path.isfile(dataset_path):  # 如果数据集文件存在
    #         dataset = torch.load(dataset_path)
    #         print(f"Dataset {i} has been loaded from {dataset_path}.")
    #
    #         # 对数据集中的每个样本进行裁剪和缩放
    #         augmented_dataset = []
    #         for image_tensor in dataset:
    #             augmented_tensor = augment_image_tensor(image_tensor, zoom_factor)
    #             augmented_dataset.append(augmented_tensor)
    #
    #         loaded_datasets.append(augmented_dataset)
    #     else:
    #         print(f"Error: Dataset {i} was not found at {dataset_path}.")
    #         # 这里可以加上异常处理或重新加载数据集的代码
    image_directory1 = "./cut_imgs/"
    image_directory2 = "./cut_images_5/"

    # 定义图片的转换操作
    transform = transforms.Compose([
        transforms.ToTensor(),
        # 可以根据需要添加其他转换操作
    ])

    # 读取图片并创建数据集
    loaded_datasets = [load_image_datasets(image_directory1), load_image_datasets(image_directory2)]

    # 合并训练数据集
    combined_dataset = torch.utils.data.ConcatDataset(loaded_datasets)
    general_data = augment_tensor_dataset(combined_dataset)

    # 加载测试数据集
    test_dataset_path_1 = './datasets/test_dataset.pt'
    test_dataset_path_2 = './datasets/test_dataset_5.pt'

    if os.path.isfile(test_dataset_path_1) and os.path.isfile(test_dataset_path_2):
        specific_data_1 = torch.load(test_dataset_path_1)
        specific_data_2 = torch.load(test_dataset_path_2)

        # 将两个测试数据集合并
        specific_data = torch.cat((specific_data_1, specific_data_2), dim=0)

        print(f"Test dataset has been loaded and merged from {test_dataset_path_1} and {test_dataset_path_2}.")

        # 对测试数据集中的每个样本进行裁剪和缩放
        augmented_specific_data = []
        for image_tensor in specific_data:
            augmented_tensor = augment_image_tensor(image_tensor, zoom_factor)
            augmented_specific_data.append(augmented_tensor)

        specific_data = augmented_specific_data

    else:
        print("Test dataset files not found.")
        specific_data = None

    return general_data, specific_data


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # general_data: 通用训练材料
    # specific_data: 指定任务的训练素材
    # 在主函数中调用 load_datasets 函数
    datasets_dir = './datasets/'
    test_dataset_dir = './datasets/test_dataset.pt'
    general_data, specific_data = load_datasets()

    general_dataset = TensorDataset(general_data)
    specific_dataset = TensorDataset(specific_data)

    general_loader = DataLoader(general_dataset, batch_size=5, shuffle=True)
    specific_loader = DataLoader(specific_dataset, batch_size=5, shuffle=True)

    # 定义MLMA模型
    mlma_model = MLMA(1e-5, 1e-8, device)

    model_path = './saved_model/VAE_cold_start.pth'
    model_dir = os.path.dirname(model_path)
    print('Try to load model from', model_path)
    # 检查模型文件夹路径是否存在
    if not os.path.exists(model_dir):
        # 不存在就创建新的目录
        os.makedirs(model_dir)
        print(f"Created directory '{model_dir}' for saving models.")
    if os.path.isfile(model_path):
        try:
            mlma_model.model.load_state_dict(torch.load(model_path, map_location=device))
            print("Model loaded successfully from '{}'".format(model_path))
        except Exception as e:
            print("Failed to load model. Starting from scratch. Error: ", e)
    else:
        print("No saved model found at '{}'. Starting from scratch.".format(model_path))

    # 定义训练循环需要的变量
    num_meta_iterations = 100  # 元迭代次数
    num_tasks = 5  # 任务数量
    num_inner_steps = 5  # 每个任务的内部更新步数
    num_samples_per_task = 10  # 每个任务采样的样本数

    # 训练循环
    for meta_iteration in range(num_meta_iterations):
        print('-' * 10, 'Meta Iteration', meta_iteration, '-' * 10)

        # 调用train_maml_vae函数进行训练
        train_maml_vae(mlma_model, general_loader, specific_loader, num_tasks, num_inner_steps, num_samples_per_task,
                       meta_iteration)

        # 保存当前的模型参数
        vae_model_state_dict = mlma_model.model.state_dict()
        torch.save(vae_model_state_dict, './saved_model/VAE_cold_start.pth')

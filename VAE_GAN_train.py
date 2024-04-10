from DataLoader import *
from GenerateTestDataset import load_test_dataset
from GAN import *
from DataAugment import *
from ImageTools import *


def augment_tensor_dataset(tensor_dataset):
    augmented_tensors = []
    for tensor in tensor_dataset:
        augmented_images = augment_image_tensor(tensor)
        # augmented_images = do_nothing(tensor)  # 什么增强都不做的函数，只进行resize
        for i in range(len(augmented_images)):
            augmented_images[i] = limit_size(augmented_images[i])
        augmented_tensors.extend(augmented_images)
    return torch.stack(augmented_tensors)  # 将列表转换回一个新的张量


def load_image_datasets(image_dir):
    image_files = os.listdir(image_dir)
    images = []
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        images.append(load_image_to_tensor(image_path))
    dataset = TensorDataset(images)
    return dataset


if __name__ == '__main__':
    loaded_datasets = []
    """从'./cut_imgs/目录中读取所有图片并加载成数据集用于训练'"""
    image_directory = "./cut_imgs/"
    # 读取图片并创建数据集
    dataset = load_image_datasets(image_directory)
    loaded_datasets.append(dataset)

    # 合并数据集
    combined_dataset = torch.utils.data.ConcatDataset(loaded_datasets)
    # 数据增强
    augmented_dataset = augment_tensor_dataset(combined_dataset)

    # 加载测试图片数据集
    # 这里的图片测试集可以由GenerateTestDataset.py中的load_test_dataset()得到
    test_dataset_path = './datasets/test_dataset.pt'
    if os.path.isfile(test_dataset_path):
        test_dataset = torch.load(test_dataset_path)
        print(f"Test dataset has been loaded from {test_dataset_path}.")
    else:
        print(f"Error: Test dataset was not found at {test_dataset_path}.")
        # 没有的时候就调用函数加载测试图片数据集
        test_dataset = load_test_dataset()

    train_epochs = 60

    model = VAEGANModelLoader(augmented_dataset, test_dataset, 10, './saved_model/VAE.pth',
                              './saved_model/Discriminator.pth')
    model.train(train_epochs, 5)
    model.test()

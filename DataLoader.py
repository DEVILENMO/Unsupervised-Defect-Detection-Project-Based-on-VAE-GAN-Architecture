import io
import zipfile

from PIL import Image
from torchvision import transforms

from Datasets import *
from ImagePreProcessing import *
from OriginalDataStructure import *


def convert_to_rgb(img):
    return img.convert('RGB')


class ZipFileLoader:
    def __init__(self, file_path: str):
        # 读取zip文件
        self.zip_file = zipfile.ZipFile(file_path, 'r')

        # 按照层级读取文件目录
        dir_structure = {}
        for file_name in self.zip_file.namelist():
            path_parts = file_name.split('/')
            current_level = dir_structure
            for part in path_parts[:-1]:
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]
            if path_parts[-1]:
                current_level[path_parts[-1]] = file_name
        # 保存结果
        self.directory = dir_structure

    def print_dir_structure(self):
        print('-' * 10, 'zip files directory', '-' * 10)
        self.__print_dir_structure(self.directory)
        print('-' * 10, 'zip files directory', '-' * 10)

    def __print_dir_structure(self, dir, indent=0):
        """递归print压缩包内的文件和目录"""
        for key, value in dir.items():
            if isinstance(value, dict):
                print('    ' * indent + key + '/')
                self.__print_dir_structure(value, indent + 1)  # 递归深度++
            else:
                print('    ' * indent + value)

    def read_file_from_zip(self, file_path):
        return self.zip_file.read(file_path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.zip_file.close()


class ZipDataLoader:
    """
    【对外提供的数据集共包含 5 类场景，每类场景包含 OK、NG 两个目录：OK 目录下
    存放 50 张 OK 图，用于训练和测试，参赛者自行确定用于训练的样本；NG 目录下
    存放 50 张 NG 图(PCB 板 7P 场景下为 20 张 NG 图)，仅用于测试，不能用于训练】
    """

    def __init__(self, dataset_path: str, debug_mode=False):
        self.debug_mode = debug_mode
        self.loaded_zip_file = ZipFileLoader(dataset_path)
        if debug_mode:
            self.loaded_zip_file.print_dir_structure()

    def load_datas(self):
        OK_datasets = []
        NG_set_list = []
        # 一级目录五个 PCB板*P
        for first_level_folder in self.loaded_zip_file.directory:
            first_level_content = self.loaded_zip_file.directory[first_level_folder]
            # 跳过readme.txt
            if isinstance(first_level_content, dict):
                if self.debug_mode:
                    print('Loading', first_level_folder)
                # 二级目录OK/NG，训练时只需要OK
                # NG set
                mark_txt = None
                label_color_txt = None
                NG_data_list = []  # list<NGData>
                # OK set
                OK_set = []  # OK是一个list<Image>，数据类型之后再改
                for second_level_folder in first_level_content:
                    second_level_content = first_level_content[second_level_folder]
                    if isinstance(second_level_content, dict):
                        if self.debug_mode:
                            print('--Loading', second_level_folder)
                        if second_level_folder == 'OK':
                            for file_name, file_path in second_level_content.items():
                                if self.debug_mode:
                                    print('  --Loading', file_name, ' - ', file_path)
                                if file_name.lower().endswith('.bmp'):
                                    image_data = self.loaded_zip_file.read_file_from_zip(file_path)
                                    image = Image.open(io.BytesIO(image_data))
                                    OK_set.append(image)
                        if second_level_folder == 'NG':
                            for file_name, file_path in second_level_content.items():
                                if self.debug_mode:
                                    print('  --Loading', file_name, ' - ', file_path)
                                if file_name.lower().endswith('_t.bmp'):
                                    continue
                                if file_name.lower().endswith('.bmp'):
                                    #         PCB░σ7p/NG/sample5.bmp
                                    image_data = self.loaded_zip_file.read_file_from_zip(file_path)
                                    image = Image.open(io.BytesIO(image_data))
                                    # image.show()
                                    #         PCB░σ7p/NG/Image_20231121194513337_t.bmp
                                    t_image_path = str(file_path).replace('.bmp', '_t.bmp')
                                    t_image_data = self.loaded_zip_file.read_file_from_zip(t_image_path)
                                    t_image = Image.open(io.BytesIO(t_image_data))
                                    #         PCB░σ7p/NG/Image_20231121194513337.xml
                                    xml_file_path = str(file_path).replace('.bmp', '.xml')
                                    xml_file_data = self.loaded_zip_file.read_file_from_zip(xml_file_path)
                                    NG_data = NGData(image, t_image, xml_file_data)
                                    NG_data_list.append(NG_data)
                                    continue
                                if file_name.lower().endswith('mark.txt'):
                                    mark_txt = self.loaded_zip_file.read_file_from_zip(file_path)
                                    continue
                                if file_name.lower().endswith('label_color.txt'):
                                    label_color_txt = self.loaded_zip_file.read_file_from_zip(file_path)
                                    continue

                print('-' * 10, first_level_folder, '-' * 10)
                print('Loaded', len(OK_set), 'OK data.')
                # augmented_OK_set = augment_images(OK_set)
                # print('Augmented to', len(augmented_OK_set), 'Ok data.')

                for index, image in enumerate(OK_set):
                    cv_image = np.array(image.convert('RGB'))
                    # 转换为BGR格式
                    cv_image = cv_image[:, :, ::-1]
                    cv_image = img_pre_processing_gray(cv_image)
                    pil_image = Image.fromarray(cv_image)
                    OK_set[index] = pil_image

                # transform = transforms.Compose([
                #     transforms.Resize((512, 512)),  # 将图像调整为512x512
                #     transforms.Lambda(convert_to_rgb),  # 确保图像为三通道
                #     transforms.ToTensor()  # 将图像转换为PyTorch张量
                # ])
                transform = transforms.Compose([
                    # transforms.Resize((512, 512)),  # 将图像调整为512x512
                    transforms.ToTensor()  # 将图像转换为PyTorch张量
                ])
                Ok_dataset = ImageDataset(images=OK_set, transform=transform)
                OK_datasets.append(Ok_dataset)

                print('Loaded', len(NG_data_list), 'NG data.')
                NG_set = NGSet(mark_txt, label_color_txt, NG_data_list)
                NG_set_list.append(NG_set)

        # 全部载入
        print('-' * 20)
        print('Loaded', len(OK_datasets), 'OK datasets.')
        print('Loaded', len(NG_set_list), 'NG datasets.')
        return OK_datasets, NG_set_list


def augment_images(image_list: list) -> list:
    augmented_list = []
    for image in image_list:
        # Original image
        augmented_list.append(image)

        # Mirror image
        mirror_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        augmented_list.append(mirror_image)

        # Rotate 90, 180, 270 degrees and mirror each of them
        for angle in [90, 180, 270]:
            rotated_image = image.rotate(angle)
            augmented_list.append(rotated_image)
            # Mirror of rotated image
            mirror_rotated_image = rotated_image.transpose(Image.FLIP_LEFT_RIGHT)
            augmented_list.append(mirror_rotated_image)

    return augmented_list

import math
import sys
import cv2
import matplotlib
import numpy
import numpy as np

from TargetDiscriminator import *
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

sys.path.append("..")

matplotlib.use('TkAgg')
crop_mode = True  # 是否裁剪到最小范围
input_dir = 'input'
output_dir = 'output'
image_files = [f for f in os.listdir(input_dir) if
               f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG', 'bmp'))]
files_num = len(image_files)

sam_checkpoint = "./saved_model/sam_vit_l_0b3195.pth"
sam_model_type = "vit_l"

device = "cuda"

sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

general_kernel1 = (100, 10)
general_kernel2 = (150, 10)


def pad_to_square(arr):
    h, w = arr.shape[:2]
    max_dim = max(h, w)

    top = (max_dim - h) // 2
    bottom = max_dim - h - top
    left = (max_dim - w) // 2
    right = max_dim - w - left

    padded_arr = np.pad(arr, ((top, bottom), (left, right), (0, 0)), mode='constant', constant_values=255)

    restore_info = {
        'original_shape': (h, w),
        'top': top,
        'left': left
    }

    return padded_arr, restore_info


def scale_image_to_fit_window(image, max_width=896, max_height=750):
    """
    将图片缩放以适应给定的最大宽度和高度。
    """
    height, width = image.shape[:2]

    # 计算缩放比例
    scale_x = max_width / width
    scale_y = max_height / height
    scale = min(scale_x, scale_y)
    # 缩放图片
    scaled_image = cv2.resize(image, None, fx=scale, fy=scale)

    return scaled_image


def apply_mask(image, mask, alpha_channel=True, kernel_size=()) -> tuple[np.ndarray, np.ndarray]:
    if (isinstance(kernel_size, tuple) or isinstance(kernel_size, list)) and len(kernel_size) == 2:
        # 将布尔类型的mask转换为uint8类型
        print('优化mask...')
        mask_processed = mask.astype(np.uint8) * 255

        # 应用腐蚀和膨胀操作
        kernel_1 = np.ones((kernel_size[0], kernel_size[0]), np.uint8)
        kernel_2 = np.ones((kernel_size[1], kernel_size[1]), np.uint8)

        mask_eroded = cv2.erode(mask_processed, kernel_1, iterations=1)

        # 对腐蚀后的图像进行连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_eroded, connectivity=8)

        # 设置连通域面积阈值
        area_threshold = 10000

        # 创建一个新的mask用于存储处理后的结果
        mask_processed = np.zeros_like(mask_eroded)

        # 只保留面积大于阈值的连通域
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= area_threshold:
                mask_processed[labels == i] = 255
        mask_dilated = cv2.dilate(mask_processed, kernel_1, iterations=1)
        mask_dilated = cv2.dilate(mask_dilated, kernel_2, iterations=1)
        mask_processed = cv2.erode(mask_dilated, kernel_2, iterations=1)

        # 显示处理前后的图像
        # scale_factor = 0.5
        # resized_original_mask = cv2.resize(mask.astype(np.uint8) * 255, None, fx=scale_factor, fy=scale_factor)
        # resized_processed_mask = cv2.resize(mask_processed, None, fx=scale_factor, fy=scale_factor)
        # cv2.imshow("Original Mask", resized_original_mask)
        # cv2.imshow("Processed Mask", resized_processed_mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 将处理后的mask转换回布尔类型
        mask = mask_processed.astype(bool)

    if alpha_channel:
        alpha = np.zeros_like(image[..., 0])  # 制作掩体
        alpha[mask] = 255  # 兴趣地方标记为1,且为白色
        image = cv2.merge((image[..., 0], image[..., 1], image[..., 2], alpha))  # 融合图像
        image = Image.fromarray(image).convert("RGBA")  # 读取图像并转换为RGB模式

        # 如果图像有alpha通道,将RGBA图像转换为灰度图,并将透明部分填充为白色
        background = Image.new("RGBA", image.size, (255, 255, 255))
        image = np.array(Image.alpha_composite(background, image))
    else:
        image = np.where(mask[..., None], image, 0)

    # 显示原图和处理后的图像
    # scale_factor = 0.5
    # resized_original_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
    # resized_processed_image = cv2.resize(np.array(image), None, fx=scale_factor, fy=scale_factor)
    # cv2.imshow("Original Image", resized_original_image)
    # cv2.imshow("Processed Image", resized_processed_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return image, mask


def get_next_filename(base_path, filename):  # 进行下一个图像
    name, ext = os.path.splitext(filename)
    for i in range(1, 101):
        new_name = f"{name}_{i}{ext}"
        if not os.path.exists(os.path.join(base_path, new_name)):
            return new_name
    return None


def save_masked_image(image, mask, output_dir, filename, crop_mode_, kernel_size):  # 保存掩盖部分的图像（感兴趣的图像）
    height, width = image.shape[:2]
    if crop_mode_:
        y, x = np.where(mask)
        y_min, y_max, x_min, x_max = y.min(), y.max(), x.min(), x.max()
        cropped_mask = mask[y_min:y_max + 1, x_min:x_max + 1]
        cropped_image = image[y_min:y_max + 1, x_min:x_max + 1]
        masked_image, cropped_mask = apply_mask(cropped_image, cropped_mask, kernel_size=kernel_size)
        masked_image, info = pad_to_square(masked_image)
        print(masked_image.shape)
    else:
        masked_image, mask = apply_mask(image, mask, kernel_size=kernel_size)
    filename = filename[:filename.rfind('.')] + '.png'
    new_filename = get_next_filename(output_dir, filename)

    if new_filename:
        if masked_image.shape[-1] == 4:
            cv2.imwrite(os.path.join(output_dir, new_filename), masked_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:
            cv2.imwrite(os.path.join(output_dir, new_filename), masked_image)
        print(f"Saved as {new_filename}")
    else:
        print("Could not save the image. Too many variations exist.")


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    return sorted_anns


def cut_target_from_image(origin_image, reason_mode='cuda', area_lower_limit=30000, area_upper_limit=math.inf,
                          pcb_prob=0.2):
    """
    Input:
            origin image: PIL image
            reason mode: cpu or gpu
            area lower limit: the min area of target object, default = 30000
            area upper limit: the max area of target object, default = positive infinity
            pcb prob: the probability of cropped image being a real PCB, default = 0.2
    Output:
            masked_image: origin size
            coordinates: (x_min, x_max, y_min, y_max), the location of the selected PCB in the original image
            cropped mask: in order to eliminate the influence of the shadow in the reconstruction
    """
    test_device = reason_mode
    sam.to(device=test_device)
    target_discriminator = TargetDiscriminator('./saved_model/Discriminator_trained.pth', device=reason_mode)
    image = numpy.array(origin_image)
    image_crop = image.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = scale_image_to_fit_window(image_rgb)
    info = None

    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)

    for j in range(len(masks)):
        if area_lower_limit < masks[j]['area'] < area_upper_limit:
            target_mask = masks[j]['segmentation']
        else:
            continue

        binary_image = np.uint8(target_mask) * 255
        resized_image = cv2.resize(binary_image, (image_crop.shape[1], image_crop.shape[0]))
        target_mask = resized_image > 0

        # masked_image = apply_mask(image_crop, target_mask)

        y, x = np.where(target_mask)
        y_min, y_max, x_min, x_max = y.min(), y.max(), x.min(), x.max()
        cropped_mask = target_mask[y_min:y_max + 1, x_min:x_max + 1]
        cropped_image = image_crop[y_min:y_max + 1, x_min:x_max + 1]
        masked_image_cut, cropped_mask = apply_mask(cropped_image, cropped_mask)
        prob_pcb = target_discriminator.predict(Image.fromarray(masked_image_cut))
        print(f'The probability of being a real PCB is: {prob_pcb:.4f}')
        if prob_pcb > pcb_prob:
            if y_max - y_min > x_max - x_min:
                diff = (y_max - y_min) - (x_max - x_min)
                if diff % 2 == 0:
                    x_max += diff // 2
                    x_min -= diff // 2
                else:
                    x_max += (diff // 2) + 1
                    x_min -= (diff // 2) - 1
            if x_max - x_min > y_max - y_min:
                diff = (x_max - x_min) - (y_max - y_min)
                if diff % 2 == 0:
                    y_max += diff // 2
                    y_min -= diff // 2
                else:
                    y_max += (diff // 2) + 1
                    y_min -= (diff // 2) - 1
            cropped_mask = target_mask[y_min:y_max + 1, x_min:x_max + 1]
            cropped_image = image_crop[y_min:y_max + 1, x_min:x_max + 1]
            masked_image_cut, cropped_mask = apply_mask(cropped_image, cropped_mask)
            break
    return Image.fromarray(masked_image_cut), [x_min, x_max, y_min, y_max], cropped_mask, info


if __name__ == '__main__':
    for i in range(files_num):
        print("第{}张图:".format(i + 1))
        filename = image_files[i]
        image = cv2.imread(os.path.join(input_dir, filename))
        image_crop = image.copy()
        # image_crop = scale_image_to_fit_window(image).copy()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = scale_image_to_fit_window(image_rgb)
        # plt.figure(figsize=(20,20))
        # plt.imshow(image)

        mask_generator = SamAutomaticMaskGenerator(sam)

        masks = mask_generator.generate(image)

        # plt.figure(figsize=(20, 20))
        # plt.imshow(image)

        masks_list = show_anns(masks)

        # 遍历分析每个预测的掩码
        for _, mask in enumerate(masks):
            print(f"Mask {_}:")
            for key, value in mask.items():
                print(f"{key}: {value}")
            print("---")

        target_mask = None
        MIN_AREA = 40000
        for j in range(len(masks_list)):
            if MIN_AREA < masks[j]['area']:
                target_mask = masks[j]['segmentation']

        if target_mask is not None:
            binary_image = np.uint8(target_mask) * 255
            resized_image = cv2.resize(binary_image, (image_crop.shape[1], image_crop.shape[0]))
            target_mask = resized_image > 0

            # masked_image = apply_mask(image_crop, target_mask)

            y, x = np.where(target_mask)
            y_min, y_max, x_min, x_max = y.min(), y.max(), x.min(), x.max()
            cropped_mask = target_mask[y_min:y_max + 1, x_min:x_max + 1]
            cropped_image = image_crop[y_min:y_max + 1, x_min:x_max + 1]
            masked_image_cut, cropped_mask = apply_mask(cropped_image, cropped_mask)

            save_masked_image(image_crop, target_mask, output_dir, filename, crop_mode_=crop_mode, kernel_size=0)
        else:
            print('未找到区域！')

        # plt.axis('off')
        # plt.show()

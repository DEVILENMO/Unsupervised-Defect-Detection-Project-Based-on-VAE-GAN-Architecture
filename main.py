import time

from AutoEncoder import *
from ImagePreProcessing import *
from ImageTools import *
from CutTarget import cut_pcb_from_image, apply_mask, pad_to_square


# 计算两个图像之间的结构相似性指数(SSIM)
def calculate_ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


# 对图像进行预处理
def preprocess_image(img):
    img = cv2.medianBlur(img, 3)  # 中值滤波去噪
    # img = cv2.equalizeHist(img)  # 直方图均衡化增强对比度
    return img


# 对差异图像进行后处理
def postprocess_diff(diff_img):
    # 高阈值二值化
    _, high_thresh_bin = cv2.threshold(diff_img, 200, 255, cv2.THRESH_BINARY)  # 调整高阈值

    # 低阈值二值化
    _, diff_bin = cv2.threshold(diff_img, 45, 255, cv2.THRESH_BINARY)  # 调整低阈值
    # _, diff_bin = cv2.threshold(diff_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态学操作
    diff_bin = cv2.dilate(diff_bin, np.ones((3, 3)), iterations=1)  # 膨胀操作
    diff_bin = cv2.erode(diff_bin, np.ones((13, 13)), iterations=1)  # 腐蚀操作
    diff_bin = cv2.dilate(diff_bin, np.ones((11, 11)), iterations=1)  # 膨胀操作

    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(diff_bin, connectivity=8)

    # 创建一个新的掩码用于存储处理后的结果
    processed_diff_bin = np.zeros_like(diff_bin)

    # 遍历每个连通域
    for i in range(1, num_labels):
        # 获取当前连通域的掩码
        component_mask = (labels == i).astype(np.uint8)

        # 检查当前连通域在高阈值二值化图像中是否存在非零像素
        if cv2.countNonZero(high_thresh_bin & component_mask) > 0:
            # 如果存在非零像素,则保留该连通域
            processed_diff_bin |= component_mask

    # scale_factor = 0.5
    # resized_high_thresh_bin = cv2.resize(high_thresh_bin, None, fx=scale_factor, fy=scale_factor)
    # resized_diff_bin = cv2.resize(diff_bin, None, fx=scale_factor, fy=scale_factor)
    # resized_processed_diff_bin = cv2.resize(processed_diff_bin, None, fx=scale_factor, fy=scale_factor)

    # cv2.imshow("High Threshold Binary", resized_high_thresh_bin)
    # cv2.imshow("Low Threshold Binary", resized_diff_bin)
    # cv2.imshow("Processed Diff Binary", resized_processed_diff_bin)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return processed_diff_bin


# 将缺陷图映射到原图中去
def resize_defection_map_to_origin_size(defection_map, mask_image, coordinates, info, original_size):
    origin_cropped_defection_map = cv2.resize(defection_map, mask_image.size)
    x_min, x_max, y_min, y_max = coordinates
    width, height = original_size

    # 创建一个和原始大图相同大小的空白图像
    origin_defection_map = np.ones((height, width), dtype=np.uint8) * 0  # 填充为黑色

    # 将小图像放置到原始大图中指定位置
    origin_defection_map[y_min:y_max + 1, x_min:x_max + 1] = origin_cropped_defection_map
    return origin_defection_map


def pcb_defection_detection(file_path, VAE_model=None, reason_mode='cuda'):
    image = Image.open(file_path)
    print('prepare to cut PCB from the image...')
    t0 = time.time()
    mask_image, coordinates, mask, info = cut_pcb_from_image(image, reason_mode=reason_mode)
    x_min, x_max, y_min, y_max = coordinates
    t1 = time.time()
    print(f'Cut PCB from image successfully, take {t1 - t0} seconds.')
    # plt.imshow(mask_image)
    # plt.show()
    device = reason_mode
    VAE_model.to(device)
    image_tensor = convert_image_to_tensor(mask_image)

    size_transform = transforms.Resize((512, 512))  # 将图像调整为512x512
    # 应用预处理转换链
    input_tensor = size_transform(image_tensor)

    # 添加批次维度并将图像输入模型
    input_tensor = input_tensor.to(device).unsqueeze(0)  # 添加批次维度,即从C x H x W变为1 x C x H x W

    t2 = time.time()
    print(f'VAE input image loaded, take {t2 - t1} seconds.')
    with torch.no_grad():
        # 编码图像,获取潜在空间的均值
        z_mu, _ = VAE_model.encoder(input_tensor)
        # 解码
        regenerated_image = VAE_model.decoder(z_mu)

    regenerated_img = regenerated_image.cpu().squeeze(0)  # 从batch中移除,得到inputchannel x 512 x 512的图片
    regenerated_img = regenerated_img.permute(1, 2, 0)  # 调整为512x512x input_channel
    regenerated_img = regenerated_img.squeeze()  # 去除单一通道维度
    # 对重建图像进行归一化处理
    regenerated_img = regenerated_img.clamp(0, 1)  # 将像素值限制在[0, 1]范围内
    t3 = time.time()
    print(f'VAE has regenerated the input image, take {t3 - t2} seconds.')

    print('Dealing with regenerated image...')
    # 将归一化后的图像乘以255,转换为[0, 255]范围内的整数值
    regenerated_img_np = regenerated_img.numpy()
    regenerated_img_np = (regenerated_img_np * 255).astype(np.uint8)
    regenerated_img_np_origin = cv2.resize(regenerated_img_np, (mask.shape[1], mask.shape[0]))
    rgb_image = cv2.cvtColor(regenerated_img_np_origin, cv2.COLOR_GRAY2RGB)
    gray_image, mask, _ = apply_mask(rgb_image, mask)
    regenerated_img_np = cv2.cvtColor(gray_image, cv2.COLOR_RGB2GRAY)
    regenerated_img_np = cv2.resize(regenerated_img_np, (512, 512))
    # plt.imshow(regenerated_img)
    # plt.show()

    # 读取并预处理原始图像和重建图像
    ori_image = load_image_with_alpha_channel(file_path)
    original_image = ori_image.copy()
    original_size = original_image.size
    original_image = np.array(original_image)
    original_image_crop, mask, _ = apply_mask(original_image[y_min:y_max + 1, x_min:x_max + 1], mask)
    print(original_image_crop.shape)
    original_image = cv2.cvtColor(original_image_crop, cv2.COLOR_RGB2GRAY)
    original_image = preprocess_image(original_image)
    reconstructed_img = regenerated_img_np
    reconstructed_img = preprocess_image(reconstructed_img)

    # 调整大小并计算SSIM
    original_image = cv2.resize(original_image, (512, 512))
    reconstructed_img = cv2.resize(reconstructed_img, (512, 512))
    ssim = calculate_ssim(original_image, reconstructed_img)
    print("Structural Similarity (SSIM) Index: ", ssim)

    # 计算差异图像并进行后处理
    diff_img = np.abs(original_image.astype(np.float32) - reconstructed_img.astype(np.float32))
    diff_img = (diff_img * 255.0 / diff_img.max()).astype(np.uint8)
    diff_bin = postprocess_diff(diff_img)

    diff_bin = resize_defection_map_to_origin_size(diff_bin, mask_image, coordinates, info, original_size)
    print(f'Total time cost: {time.time() - t0} seconds.')

    # 显示结果
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(ori_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 4, 2)
    plt.imshow(reconstructed_img, cmap='gray')
    plt.title('Reconstructed Image')
    plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.imshow(diff_img, cmap='jet')
    plt.title('Difference Image')
    plt.axis('off')
    plt.subplot(1, 4, 4)
    plt.imshow(diff_bin, cmap='gray')
    plt.title('Defect Regions')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return ssim, diff_bin


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 初始化模型
    encoder = VAEEncoder(latent_dim)
    # decoder
    decoder = VAEDecoder(latent_dim)
    # VAE
    VAE_model = VAEModel(encoder, decoder).to(device)
    model_path = './saved_model/VAE.pth'
    VAE_model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully from '{}'".format(model_path))

    input_dir = r'input'
    output_dir = r'output'
    image_files = [f for f in os.listdir(input_dir) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG', 'bmp'))]
    files_num = len(image_files)

    # 创建一个空列表用于存储图片对应关系
    image_pairs = []

    for i in range(files_num):
        filename = image_files[i]
        print("第{}张图:".format(i + 1), filename)

        # 判断文件名是否包含'bin',如果包含则跳过
        if 'bin' in filename or '_t.' in filename:
            print("跳过文件: {}".format(filename))
            continue

        score, diff_bin = pcb_defection_detection(os.path.join(input_dir, filename), VAE_model=VAEModel,
                                                 reason_mode=str(device))

        # 打印返回的分数
        print("相似度:", score)

        # 对二值化图像进行反色处理
        diff_bin = cv2.bitwise_not(diff_bin)

        # 将二值化后的图像保存到输出路径
        filename_without_ext = os.path.splitext(filename)[0]
        # 构建新的文件名
        new_filename = f"{filename_without_ext}_bin.png"
        output_path = os.path.join(output_dir, new_filename)
        cv2.imwrite(output_path, diff_bin)

        # 将输入图片文件名和输出图片文件名加入到图片对应关系列表中
        image_pairs.append(f"{filename} {new_filename}")

    # 将图片对应关系写入txt文件
    txt_file = "./output/mark.txt"

    # 检查文件是否存在
    if os.path.exists(txt_file):
        # 如果文件存在,则删除文件
        os.remove(txt_file)

    # 将新的图片对应关系写入txt文件
    with open(txt_file, "w") as file:
        file.write("\n".join(image_pairs))

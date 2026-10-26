"""全局配置: 项目路径与常量统一在此定义, 不要在业务代码里硬编码路径。"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- 数据目录
INPUT_DIR = PROJECT_ROOT / "input"            # 待检测原图
OUTPUT_DIR = PROJECT_ROOT / "output"          # 检测结果输出
TRAIN_IMAGE_DIR = PROJECT_ROOT / "train_images"   # 训练用正常样本 (SAM 切割后)
TEST_IMAGE_DIR = PROJECT_ROOT / "test_images"     # 留出测试样本, 用于生成 test_dataset.pt
CUT_IMAGE_DIR = PROJECT_ROOT / "cut_imgs"         # SAM 批量切割输出
REGEN_VIEW_DIR = PROJECT_ROOT / "test_imgs"       # 训练过程中用于重建可视化的图片

# ---------------------------------------------------------------- 模型权重
SAVED_MODEL_DIR = PROJECT_ROOT / "saved_model"
VAE_CHECKPOINT = SAVED_MODEL_DIR / "VAE.pth"
VAE_BEST_CHECKPOINT = SAVED_MODEL_DIR / "VAE_best.pth"
VAE_BACKUP_CHECKPOINT = SAVED_MODEL_DIR / "VAE_backup.pth"
VAE_COLD_START_CHECKPOINT = SAVED_MODEL_DIR / "VAE_cold_start.pth"
DISCRIMINATOR_CHECKPOINT = SAVED_MODEL_DIR / "Discriminator.pth"
DISCRIMINATOR_TRAINED_CHECKPOINT = SAVED_MODEL_DIR / "Discriminator_trained.pth"
SAM_CHECKPOINT = SAVED_MODEL_DIR / "sam_vit_l_0b3195.pth"
SAM_MODEL_TYPE = "vit_l"

# ---------------------------------------------------------------- 数据缓存与训练产物
DATASET_DIR = PROJECT_ROOT / "datasets"
TEST_DATASET_FILE = DATASET_DIR / "test_dataset.pt"
FIGURE_DIR = PROJECT_ROOT / "figure"          # 训练损失曲线
TRAIN_RESULT_DIR = PROJECT_ROOT / "train_result"  # VAE-GAN 训练重建结果
VAE_TEST_DIR = PROJECT_ROOT / "VAE_test"      # VAE 重建/随机生成结果

# ---------------------------------------------------------------- 模型结构常量
# 注意: 这些值决定网络层的形状, 与已保存的 checkpoint 对应, 修改后旧权重将无法加载。
IMAGE_SIZE = 512        # 网络输入/输出边长
LATENT_DIM = 256        # 潜空间维度
INPUT_CHANNEL = 1       # 输入通道数 (灰度)
BASE_CHANNEL = 1792     # 编码器最深层 / 解码器起始层的通道宽度

# 支持的图片扩展名 (统一小写比较)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")

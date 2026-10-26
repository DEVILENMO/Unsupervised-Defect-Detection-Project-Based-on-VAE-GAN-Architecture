"""ModelLoader: 训练器的通用基类, 负责数据迭代器、训练循环骨架、权重保存/加载/备份。"""

import os
import time

import torch
from torch.utils.data import DataLoader, random_split


class ModelLoader:
    def __init__(self, train_dataset, test_dataset, batch_size: int, model_path: str,
                 if_early_stop: bool = False, debug_mode: bool = False):
        """
        Args:
            train_dataset: 训练集; 与 test_dataset 同时为 None 时进入纯推理模式。
            test_dataset: 测试集; 为 None 时从训练集随机切出 1% 作为测试集。
            batch_size: 批大小。
            model_path: 权重保存/加载路径。
            if_early_stop: 是否启用早停。
            debug_mode: 调试模式 (同步 CUDA 调用 + 异常检测, 训练更慢)。
        """
        self.predict_mode = False
        if train_dataset is None and test_dataset is None:
            print("Model will run in predict mode.")
            self.predict_mode = True
        elif train_dataset is not None and test_dataset is not None:
            self.train_dataset = train_dataset
            self.test_dataset = test_dataset
            self.batch_size = batch_size
            self.train_iterator = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                             shuffle=True, drop_last=True)
            self.test_iterator = DataLoader(self.test_dataset, batch_size=self.batch_size,
                                            drop_last=True)
        else:
            print("Generate test dataset randomly.")
            test_ratio = 0.01
            train_size = int(len(train_dataset) * (1 - test_ratio))
            test_size = len(train_dataset) - train_size
            self.batch_size = batch_size
            self.train_dataset, self.test_dataset = random_split(
                train_dataset, [train_size, test_size])
            self.train_iterator = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                             shuffle=True, drop_last=True)
            self.test_iterator = DataLoader(self.test_dataset, batch_size=self.batch_size,
                                            drop_last=True)

        self.if_early_stop = if_early_stop
        self.debug_mode = debug_mode
        if debug_mode:
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            torch.autograd.set_detect_anomaly(True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = None
        self.lr = None
        self.optimizer = None
        self.scheduler = None

        self.best_loss = float("inf")
        self.train_losses = []
        self.test_losses = []
        self.current_epoch = 0

    # ------------------------------------------------------------ 子类接口

    def _train_epoch(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def _test_epoch(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def test(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def train(self, epochs: int = 50, test_interval: int = 1, figure_interval: int = 10,
              backup_interval: int = 10):
        """通用训练循环; 各子类可按需覆盖。"""
        if self.predict_mode:
            print("No data given, model is running in predict mode.")
            return
        print("Start training...")
        if self.if_early_stop:
            best_test_loss = float("inf")
            patience_counter = 0

        from defect_detection.config import FIGURE_DIR
        os.makedirs(FIGURE_DIR, exist_ok=True)

        import matplotlib.pyplot as plt

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            train_loss = self._train_epoch()
            test_loss = self._test_epoch()
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.5f}, "
                  f"Test Loss: {test_loss:.5f}")

            if self.scheduler:
                self.scheduler.step()

            if epoch % backup_interval == 0:
                self.save_model(test_loss, create_backup=True)
            else:
                self.save_model(test_loss, create_backup=False)

            if epoch % figure_interval == 0:
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, epoch + 1), self.train_losses, label="Train Loss")
                plt.plot(range(1, epoch + 1), self.test_losses, label="Test Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"Training and Test Loss - Epoch {epoch}")
                plt.legend()
                plt.grid(True)
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                model_stem = os.path.splitext(os.path.basename(self.model_path))[0]
                plt.savefig(FIGURE_DIR / f"{model_stem}-{timestamp}-epoch{epoch}.png")
                plt.close()

            if epoch % test_interval == 0:
                self.test()

            if self.if_early_stop:
                if best_test_loss > test_loss:
                    best_test_loss = test_loss
                    patience_counter = 1
                else:
                    patience_counter += 1
                if patience_counter > max(epochs / 5, 10):
                    print("Training interrupted to avoid overfitting.")
                    break

    # ------------------------------------------------------------ 权重管理

    def _save_model(self, save_path, is_best: bool = False, is_backup: bool = False):
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            print(f"Created directory: {save_dir}")

        torch.save(self.model.state_dict(), save_path)

        if is_best:
            print(f"Best model saved to {save_path}")
        elif is_backup:
            print(f"Backup model saved to {save_path}")
        else:
            print(f"Current model saved to {save_path}")

    def save_model(self, loss: float = float("inf"), create_backup: bool = False):
        """保存当前权重; 损失创新低时额外保存 _best, 可选保存 _backup。"""
        model_name, model_extension = os.path.splitext(self.model_path)

        self._save_model(self.model_path)

        if loss < self.best_loss:
            self.best_loss = loss
            self._save_model(f"{model_name}_best{model_extension}", is_best=True)

        if create_backup:
            self._save_model(f"{model_name}_backup{model_extension}", is_backup=True)

    def load_model(self):
        model_dir = os.path.dirname(self.model_path)
        print("Try to load model from", self.model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            print(f"Created directory '{model_dir}' for saving models.")
        if os.path.isfile(self.model_path):
            try:
                self.model.load_state_dict(
                    torch.load(self.model_path, map_location=self.device, weights_only=True))
                print("Model loaded successfully from '{}'".format(self.model_path))
            except Exception as e:
                print("Failed to load model. Starting from scratch. Error: ", e)
        else:
            print("No saved model found at '{}'. Starting from scratch.".format(self.model_path))

    def run(self, train_epochs: int = 50):
        """训练 + 异常兜底保存。"""
        try:
            self.train(train_epochs)
        except KeyboardInterrupt:
            print("Training interrupted by the user.")
        finally:
            self.save_model()

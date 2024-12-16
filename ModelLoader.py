import os
import time

import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt


class ModelLoader:
    def __init__(self, train_dataset, test_dataset, batch_size, model_path: str, if_early_stop=False, debug_mode=False):
        """
        初始化模型加载器。

        Args:
            train_dataset: 用于训练的数据集。
            test_dataset: 用于测试的数据集。
            batch_size: 每批数据的大小。
            model_path: 保存或加载模型权重的路径。
            if_early_stop: 是否早停。
            debug_mode: 是否为调试模式，调试模式下可能会启用额外的日志或检查点。
        """
        self.predict_mode = False
        if train_dataset is None and test_dataset is None:
            print('Model will run in predict mode.')
            self.predict_mode = True
        elif train_dataset is not None and test_dataset is not None:
            self.train_dataset = train_dataset
            self.test_dataset = test_dataset
            self.batch_size = batch_size
            self.train_iterator = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
            self.test_iterator = DataLoader(self.test_dataset, batch_size=self.batch_size, drop_last=True)
        elif train_dataset is not None and test_dataset is None:
            print('Generate test dataset randomly.')
            # 如果没有提供测试集,则从训练集中随机选择一部分作为测试集
            test_ratio = 0.01
            train_size = int(len(train_dataset) * (1 - test_ratio))
            test_size = len(train_dataset) - train_size
            self.batch_size = batch_size
            self.train_dataset, self.test_dataset = random_split(train_dataset, [train_size, test_size])
            self.train_iterator = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
            self.test_iterator = DataLoader(self.test_dataset, batch_size=self.batch_size, drop_last=True)

        self.if_early_stop = if_early_stop
        self.debug_mode = debug_mode
        if debug_mode:
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            torch.autograd.set_detect_anomaly(True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.lr = None
        self.optimizer = None
        self.scheduler = None

        self.best_loss = float('inf')
        self.train_losses = []
        self.test_losses = []
        self.current_epoch = 0

    def _train_epoch(self):
        """
        训练模型一个epoch，子类需要根据具体的模型实现该方法。
        """
        raise NotImplementedError('Subclasses should implement this method.')

    def _test_epoch(self):
        """
        测试模型一个epoch，子类需要根据具体的模型实现该方法。
        """
        raise NotImplementedError('Subclasses should implement this method.')

    def train(self, epochs=50, test_interval=1, figure_interval=10, backup_interval=10):
        """
        训练模型，周期性地在测试集上评估性能。

        Args:
            epochs: 训练的总轮次。
            test_interval: 测试间隔。
        """
        if self.predict_mode:
            print('No data given, model is running in predict mode.')
            return
        print('Start training...')
        if self.if_early_stop:
            # 早停策略防止过拟合
            best_test_loss = float('inf')
            patience_counter = 0

        figure_dir = './figure'
        os.makedirs(figure_dir, exist_ok=True)

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            train_loss = self._train_epoch()
            test_loss = self._test_epoch()
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            print(f'Epoch {epoch}/{epochs} - Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f}')

            if self.scheduler:
                self.scheduler.step()

            if epoch % backup_interval == 0:
                self.save_model(test_loss, create_backup=True)
            else:
                self.save_model(test_loss, create_backup=False)

            if epoch % figure_interval == 0:
                # 绘制损失曲线
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, epoch + 1), self.train_losses, label='Train Loss')
                plt.plot(range(1, epoch + 1), self.test_losses, label='Test Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'Training and Test Loss - Epoch {epoch}')
                plt.legend()
                plt.grid(True)
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                plt.savefig(
                    f'{figure_dir}/{os.path.splitext(os.path.basename(self.model_path))[0]}-{timestamp}-epoch{epoch}.png')
                plt.close()

            if epoch % test_interval == 0:
                self.test()

            if self.if_early_stop:
                # 计算早停累计
                if best_test_loss > test_loss:
                    best_test_loss = test_loss
                    patience_counter = 1
                else:
                    patience_counter += 1
                if patience_counter > max(epochs / 5, 10):
                    # 早停
                    print('Training interrupted to avoid overfitting.')
                    break

    def test(self):
        raise NotImplementedError('Subclasses should implement this method.')

    def _save_model(self, save_path: str, is_best: bool = False, is_backup: bool = False):
        """
        保存模型的底层实现。

        Args:
            save_path: 保存模型的完整路径
            is_best: 是否是最佳模型
            is_backup: 是否是备份模型
        """
        # 确保保存目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            print(f'Created directory: {save_dir}')

        # 保存模型
        torch.save(self.model.state_dict(), save_path)

        # 根据保存类型打印相应信息
        if is_best:
            print(f'Best model saved to {save_path}')
        elif is_backup:
            print(f'Backup model saved to {save_path}')
        else:
            print(f'Current model saved to {save_path}')

    def save_model(self, loss=float('inf'), create_backup=False):
        """
        保存模型的权重，包括当前模型、最佳模型和可选的备份模型。

        Args:
            loss: 当前损失值
            create_backup: 是否创建备份模型
        """
        model_name, model_extension = os.path.splitext(self.model_path)

        # 保存当前模型
        self._save_model(self.model_path)

        # 如果是最佳模型，额外保存一份
        if loss < self.best_loss:
            self.best_loss = loss
            best_model_path = f"{model_name}_best{model_extension}"
            self._save_model(best_model_path, is_best=True)

        # 根据参数决定是否创建备份
        if create_backup:
            backup_model_path = f"{model_name}_backup{model_extension}"
            self._save_model(backup_model_path, is_backup=True)

    def load_model(self):
        """
        加载模型的权重。
        """
        # load model weight
        model_dir = os.path.dirname(self.model_path)
        print('Try to load model from', self.model_path)
        # 检查模型文件夹路径是否存在
        if not os.path.exists(model_dir):
            # 不存在就创建新的目录
            os.makedirs(model_dir, exist_ok=True)
            print(f"Created directory '{model_dir}' for saving models.")
        if os.path.isfile(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                print("Model loaded successfully from '{}'".format(self.model_path))
            except Exception as e:
                print("Failed to load model. Starting from scratch. Error: ", e)
        else:
            print("No saved model found at '{}'. Starting from scratch.".format(self.model_path))

    def run(self, train_epochs=50):
        """
        执行训练和测试周期。
        """
        try:
            self.train(train_epochs)
        except KeyboardInterrupt:
            print('Training interrupted by the user.')
        finally:
            self.save_model()

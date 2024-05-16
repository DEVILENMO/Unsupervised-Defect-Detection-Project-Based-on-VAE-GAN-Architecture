from torch.utils.data import Dataset


class TensorDataset(Dataset):
    def __init__(self, images, transform=None):
        """
        Args:
            images (list): List of PIL images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        return image

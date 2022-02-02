"""Load the images and build a dataset for training.
"""
import os

import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms



class AnimeDataset(Dataset):
    """Custom dataset. Loads images and apply the given transforms.
    """
    def __init__(self, paths: list, transform: transforms.Compose):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.FloatTensor:
        path = self.paths[index]
        image = Image.open(path)
        if image is None:
            print('ERROR: image is None:', path)

        image = self.transform(image)
        return image


def load_dataset(path_dir: str, image_size: int) -> AnimeDataset:
    """Load the dataset with the right transforms.
    """
    paths = [
        os.path.join(path_dir, p)
        for p in os.listdir(path_dir)
    ]

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Pixels are between [0, 1].
        transforms.Lambda(lambda t: 2 * t - 1),  # Normalise between [-1, 1].
    ])

    return AnimeDataset(paths, transform)


def plot_image(image: torch.FloatTensor):
    """Plot a tensor image.
    """
    image = image.cpu().permute(1, 2, 0)
    image = image.numpy()

    plt.imshow(image)
    plt.show()

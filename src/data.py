"""Load the images and build a dataset for training.
"""
import os

import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms



class AnimeDataset(Dataset):
    def __init__(self, paths: list, transform: transforms.Compose):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        image = Image.open(path)
        if image is None:
            print('ERROR: image is None:', path)

        image = self.transform(image)
        return image


def load_dataset(path_dir: list, image_size: int):
    paths = [
        os.path.join(path_dir, p)
        for p in os.listdir(path_dir)
    ]

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    return AnimeDataset(paths, transform)


def plot_image(image: torch.FloatTensor):
    image = image.cpu().permute(1, 2, 0)
    image = image.numpy()

    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    path_dir = '../data'
    image_size = 64

    dataset = load_dataset(path_dir, image_size)
    image = dataset[0]
    print(f'Dataset of {len(dataset):,} images.')
    plot_image(image)

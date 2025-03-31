import os
import torch
from torchvision import datasets, transforms
from logic.common import DEFAULT_BATCH_SIZE, PATH_DATA


def dataset_exists(root: str, name: str) -> bool:
    return os.path.exists(os.path.join(root, name, "processed", "training.pt"))


def load_mnist(root: str = PATH_DATA) -> datasets.MNIST:
    download = not dataset_exists(root, "MNIST")
    return datasets.MNIST(root, train=True, transform=transforms.ToTensor(), download=download)


def load_fashion_mnist(root: str = PATH_DATA) -> datasets.FashionMNIST:
    download = not dataset_exists(root, "FashionMNIST")
    return datasets.FashionMNIST(root, train=True, transform=transforms.ToTensor(), download=download)


def make_dataloader(train_dataset: torch.utils.data.Dataset) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        train_dataset, batch_size=DEFAULT_BATCH_SIZE, shuffle=True
    )

import os
from torchvision import datasets, transforms
from logic.common import PATH_DATA


def dataset_exists(root: str, name: str) -> bool:
    return os.path.exists(os.path.join(root, name, "processed", "training.pt"))


def load_mnist(root: str = PATH_DATA) -> datasets.MNIST:
    download = not dataset_exists(root, "MNIST")
    return datasets.MNIST(root, train=True, transform=transforms.ToTensor(), download=download)


def load_fashion_mnist(root: str = PATH_DATA) -> datasets.FashionMNIST:
    download = not dataset_exists(root, "FashionMNIST")
    return datasets.FashionMNIST(root, train=True, transform=transforms.ToTensor(), download=download)

import os
import torch
from torchvision import datasets, transforms
from logic.common import DEFAULT_BATCH_SIZE, PATH_DATA


def dataset_exists(root: str, name: str) -> bool:
    return os.path.exists(os.path.join(root, name, "processed", "training.pt"))


def load_mnist(root: str = PATH_DATA, train: bool = True) -> datasets.MNIST:
    download = not dataset_exists(root, "MNIST")
    return datasets.MNIST(root, train=train, transform=transforms.ToTensor(), download=download)


def load_fashion_mnist(root: str = PATH_DATA, train: bool = True) -> datasets.FashionMNIST:
    download = not dataset_exists(root, "FashionMNIST")
    return datasets.FashionMNIST(root, train=train, transform=transforms.ToTensor(), download=download)

#########################################


def make_dataloader(dataset: torch.utils.data.Dataset, shuffle: bool = True) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(dataset, batch_size=DEFAULT_BATCH_SIZE, shuffle=shuffle)

###################################################################################


def get_train_dataloader(dataset_name: str) -> torch.utils.data.DataLoader:
    if dataset_name.lower() == "fashionmnist":
        ds = load_fashion_mnist(train=True)
    else:
        ds = load_mnist(train=True)
    return make_dataloader(ds, shuffle=True)


def get_test_dataloader(dataset_name: str) -> torch.utils.data.DataLoader:
    if dataset_name.lower() == "fashionmnist":
        ds = load_fashion_mnist(train=False)
    else:
        ds = load_mnist(train=False)
    return make_dataloader(ds, shuffle=False)

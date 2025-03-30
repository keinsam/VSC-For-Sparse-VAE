import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from logic.common import DEFAULT_EPOCHS, PATH_VAE, PATH_AUTOENCODER, PATH_VSC
from logic.model.vae import VAE
from logic.model.autoencoder import Autoencoder
from logic.model.vsc import VSC
from logic.train.train_vae import process_vae
from logic.train.train_autoencoder import process_autoencoder
from logic.train.train_vsc import process_vsc
from logic.train.base import display_history


def get_dataloader() -> DataLoader:
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(
        root="data/MNIST", train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=128, shuffle=True)


def run_training(name: str, process_fn, model: torch.nn.Module, model_path: str, dataloader: DataLoader, device: torch.device, epochs: int, no_cache: bool, silent: bool) -> torch.nn.Module:
    if not silent:
        print(f"Training {name}...")
    history, trained_model = process_fn(
        model, dataloader, device, no_cache, model_path, epochs)
    if not silent and history:
        display_history(history)
    return trained_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="vae,autoencoder,vsc",
                        help="Comma separated list of models to train: vae, autoencoder, vsc")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--silent", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader()
    selected = [m.strip().lower() for m in args.models.split(",")]

    if "autoencoder" in selected:
        model = Autoencoder().to(device)
        run_training("Autoencoder", process_autoencoder, model, PATH_AUTOENCODER,
                     dataloader, device, args.epochs, args.no_cache, args.silent)
    if "vae" in selected:
        model = VAE().to(device)
        run_training("VAE", process_vae, model, PATH_VAE, dataloader,
                     device, args.epochs, args.no_cache, args.silent)
    if "vsc" in selected:
        model = VSC().to(device)
        run_training("VSC", process_vsc, model, PATH_VSC, dataloader,
                     device, args.epochs, args.no_cache, args.silent)


if __name__ == '__main__':
    main()

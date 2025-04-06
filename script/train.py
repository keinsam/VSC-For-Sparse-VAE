import argparse
from typing import Dict, List, Tuple
import torch
from torch.utils.data import DataLoader
from logic.common import DEFAULT_EPOCHS, PATH_VAE, PATH_AE, PATH_VSC, PATH_VSC_WARMUP
from logic.data import get_train_dataloader
from logic.model.autoencoder import Autoencoder
from logic.model.vae import VAE
from logic.model.vsc import VSC
from logic.train.train_vae import process_vae
from logic.train.train_autoencoder import process_autoencoder
from logic.train.train_vsc import process_vsc, process_vsc_warmup
from logic.train.base import display_history
from graphics.history import visualize_history
import csv
import plotly.graph_objects as go


def run_training(
        name: str, process_fn,
        model: torch.nn.Module,
        model_path: str, dataloader: DataLoader,
        device: torch.device, epochs: int,
        no_cache: bool,
        silent: bool,
        show_history: bool,
) -> torch.nn.Module:
    if not silent:
        print(f"Training {name}...")
    history, trained_model = process_fn(
        model, dataloader, device, no_cache, model_path, epochs)

    figure: go.Figure = visualize_history(history)
    if show_history:
        figure.show()

    model_path_without_extension = model_path.rsplit('.', 1)[0]
    csv_path = f"{model_path_without_extension}_training_history.csv"
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    figure.write_image(f"{model_path_without_extension}_training_history.png")

    return trained_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="vae,autoencoder,vsc,vsc_warmup",
                        help="Comma separated list of models to train: vae, autoencoder, vsc")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--no_cache", action="store_true",
                        help="ie. force training")
    parser.add_argument("--silent", action="store_true")
    parser.add_argument("--show_history", action="store_true")
    parser.add_argument("--dataset", type=str, default="mnist",
                        help="Dataset to use: mnist or fashionmnist")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_train_dataloader(args.dataset)
    selected = [m.strip().lower() for m in args.models.split(",")]

    models_to_train = [
        (["autoencoder", "ae"], Autoencoder, process_autoencoder, PATH_AE),
        (["vae"], VAE, process_vae, PATH_VAE),
        (["vsc"], VSC, process_vsc, PATH_VSC),
        (["vsc_warmup"], VSC, process_vsc_warmup, PATH_VSC_WARMUP)
    ]

    for names, model_class, process_fn, model_path in models_to_train:
        if any(name in selected for name in names):
            model = model_class().to(device)
            run_training(names[0].capitalize(), process_fn, model, model_path,
                         dataloader, device, args.epochs, args.no_cache, args.silent,
                         args.show_history)


if __name__ == '__main__':
    main()

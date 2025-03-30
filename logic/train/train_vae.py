import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from logic.model.vae import VAE
from typing import List
import os
import torch
from torch.nn.functional import binary_cross_entropy
from common import PATH_OUTPUT, DEFAULT_EPOCHS, PATH_VAE
from logic.train.base import display_history, process


def train_vae(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    epochs: int = DEFAULT_EPOCHS,
    device: torch.device = None,
    verbose: bool = True
) -> List[dict]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    history = []
    for epoch in range(epochs):
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        for batch_idx, (x, _) in enumerate(dataloader):
            x = x.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar = model(x)
            # Reconstruction loss: sum over pixels, mean over batch
            recon_loss = F.binary_cross_entropy(
                x_recon, x, reduction='none').sum(dim=[1, 2, 3]).mean()
            kl_loss = -0.5 * (1 + logvar - mu.pow(2) -
                              logvar.exp()).sum(dim=1).mean()
            loss = recon_loss + kl_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
        avg_loss = total_loss / len(dataloader)
        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_kl_loss = total_kl_loss / len(dataloader)
        history.append({
            'epoch': epoch + 1,
            'avg_loss': avg_loss,
            'avg_recon_loss': avg_recon_loss,
            'avg_kl_loss': avg_kl_loss
        })
        if verbose:
            print(
                f'Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}')
    return history


def process_vae(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    no_cache: bool = False,
    model_path: str = PATH_VAE,
    epochs: int = DEFAULT_EPOCHS
) -> tuple:
    return process(model_path, train_vae, model, dataloader, device, no_cache, epochs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--silent", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE().to(device)
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(
        root="data/MNIST", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    history, model = process(model, dataloader, device,
                             args.no_cache, args.epochs)
    if not args.silent:
        display_history(history)


if __name__ == '__main__':
    main()

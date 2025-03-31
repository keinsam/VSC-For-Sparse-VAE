import torch
from torch.utils.data import DataLoader
from typing import List, Tuple
import torch.nn.functional as F
from logic.train.base import process
from logic.common import PATH_VAE, DEFAULT_EPOCHS


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


def process_vae(model: torch.nn.Module, dataloader: DataLoader, device: torch.device, no_cache: bool = False, model_path: str = PATH_VAE, epochs: int = DEFAULT_EPOCHS) -> Tuple[List[dict], torch.nn.Module]:
    return process(model_path, train_vae, model, dataloader, device, no_cache, epochs)

import torch
from torch.utils.data import DataLoader
from typing import List, Tuple
from logic.train.base import process
from logic.common import PATH_VSC, DEFAULT_EPOCHS


def train_vsc(model: torch.nn.Module, dataloader: DataLoader, epochs: int, device: torch.device) -> List[dict]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    history = []
    for epoch in range(epochs):
        total_loss = 0
        for x, _ in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar = model(x)
            recon_loss = torch.nn.functional.binary_cross_entropy(
                x_recon, x, reduction='none').sum(dim=[1, 2, 3]).mean()
            kl_loss = -0.5 * (1 + logvar - mu.pow(2) -
                              logvar.exp()).sum(dim=1).mean()
            sparsity_loss = model.compute_sparsity_loss(mu)
            loss = recon_loss + kl_loss + sparsity_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        history.append({'epoch': epoch + 1, 'avg_loss': avg_loss})
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    return history


def process_vsc(model: torch.nn.Module, dataloader: DataLoader, device: torch.device, no_cache: bool = False, model_path: str = PATH_VSC, epochs: int = DEFAULT_EPOCHS) -> Tuple[List[dict], torch.nn.Module]:
    return process(model_path, train_vsc, model, dataloader, device, no_cache, epochs)

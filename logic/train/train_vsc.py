from multiprocessing import process
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Tuple
# On suppose que le modèle VSC implémente compute_sparsity_loss correctement
from logic.common import DEFAULT_EPOCHS, PATH_VSC, PATH_VSC_WARMUP
from logic.model.vsc import VSC


def train_vsc_standard(model: nn.Module, dataloader: DataLoader, epochs: int, device: torch.device) -> List[dict]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    history = []
    for epoch in range(epochs):
        total_loss = 0.0
        for x, _ in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar, gamma = model(x)
            # Reconstruction loss: binary cross-entropy, somme sur pixels, moyenne sur batch.
            recon_loss = nn.functional.binary_cross_entropy(
                x_recon, x, reduction='none').sum(dim=[1, 2, 3]).mean()
            # KL divergence pour la partie continue (slab).
            kl_loss = -0.5 * (1 + logvar - mu.pow(2) -
                              torch.exp(logvar)).sum(dim=1).mean()
            # Pénalité de sparsité (doit être calculée via divergence KL entre Bernoulli(gamma) et Bernoulli(alpha)).
            sparsity_loss = model.compute_sparsity_loss(gamma)
            loss = recon_loss + kl_loss + sparsity_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(
            f"Standard Training Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        history.append({'epoch': epoch+1, 'avg_loss': avg_loss})
    return history


def train_vsc_warmup(model: VSC, dataloader: DataLoader, epochs: int, device: torch.device,
                     n_warmup: int, delta_lambda: float, L: int = 1) -> List[dict]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    history = []
    # Initialisation de lambda pour la phase de warm-up.
    model.lambda_val = 0.0
    iteration = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, _ in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            # Calcul de la loss de reconstruction sur L échantillons.
            recon_losses = []
            for _ in range(L):
                x_recon, mu, logvar, gamma = model(x)
                recon_loss = nn.functional.binary_cross_entropy(
                    x_recon, x, reduction='none').sum(dim=[1, 2, 3])
                recon_losses.append(recon_loss)
            recon_loss = torch.stack(recon_losses, dim=0).mean()
            kl_loss = -0.5 * (1 + logvar - mu.pow(2) -
                              torch.exp(logvar)).sum(dim=1).mean()
            sparsity_loss = model.compute_sparsity_loss(gamma)
            loss = recon_loss + kl_loss + sparsity_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # Warm-up : on augmente lambda jusqu'à n_warmup itérations.
            if iteration < n_warmup:
                model.lambda_val = min(1.0, model.lambda_val + delta_lambda)
            iteration += 1
        avg_loss = epoch_loss / len(dataloader)
        print(
            f"Warmup Training Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, lambda: {model.lambda_val:.2f}")
        history.append({'epoch': epoch+1, 'avg_loss': avg_loss,
                       'lambda': model.lambda_val})
    return history


def process_vsc(model: VSC, dataloader: DataLoader, device: torch.device,
                no_cache: bool = False, epochs: int = DEFAULT_EPOCHS) -> Tuple[List[dict], VSC]:
    # Utilisation de la fonction standard de training pour VSC
    return process(PATH_VSC, train_vsc_standard, model, dataloader, device, no_cache, epochs)


def process_vsc_warmup(model: VSC, dataloader: DataLoader, device: torch.device,
                       no_cache: bool = False, epochs: int = DEFAULT_EPOCHS,
                       n_warmup: int = 100, delta_lambda: float = 0.01, L: int = 1) -> Tuple[List[dict], VSC]:
    # Définir une fonction lambda pour passer les paramètres de warm-up
    def training_function(m, d, e, dev): return train_vsc_warmup(
        m, d, e, dev, n_warmup, delta_lambda, L)
    return process(PATH_VSC_WARMUP, training_function, model, dataloader, device, no_cache, epochs)

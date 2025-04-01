import os
import torch
from typing import List, Callable, Tuple
from logic.common import DEFAULT_EPOCHS
from logic.model.base import load_model


def process(
    model_path: str,
    training_function: Callable[[torch.nn.Module, torch.utils.data.DataLoader, int, torch.device], List[dict]],
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    no_cache: bool = False,
    epochs: int = DEFAULT_EPOCHS
) -> Tuple[List[dict], torch.nn.Module]:
    if not no_cache and os.path.exists(model_path):
        model, history = load_model(model, model_path)
        return history, model
    history = training_function(model, dataloader, epochs, device)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    return history, model


def display_history(history: List[dict]) -> None:
    for record in history:
        print(f"Epoch {record['epoch']}, Avg Loss: {record['avg_loss']:.4f}" +
              (f", Recon Loss: {record.get('avg_recon_loss', 0):.4f}, KL Loss: {record.get('avg_kl_loss', 0):.4f}" if 'avg_recon_loss' in record else ""))

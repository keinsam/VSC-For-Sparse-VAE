import os
from typing import Dict, List, Tuple
import torch
from logic.common import PATH_VAE
from logic.utils import get_device
import csv


def load_model(
        model: torch.nn.Module,
        model_path: str = PATH_VAE,
        device: torch.device = None
) -> torch.nn.Module:
    if device is None:
        device = get_device()
    if os.path.exists(model_path):
        state_dict = torch.load(
            model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        return None
    return None


def load_model_with_history(
        model: torch.nn.Module,
        model_path: str = PATH_VAE,
        device: torch.device = None
) -> Tuple[torch.nn.Module, List[Dict]]:

    model = load_model(model, model_path, device)
    if model is None:
        return None

    # Load training history
    model_path_without_extension = model_path.rsplit('.', 1)[0]
    csv_path = f"{model_path_without_extension}_training_history.csv"

    history = []
    if os.path.exists(csv_path):
        with open(csv_path, mode='r') as csv_file:
            reader = csv.DictReader(csv_file)
            history = [row for row in reader]

    return model, history

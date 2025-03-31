import os
import torch
from logic.common import PATH_VAE
from logic.utils import get_device


def load_model(model: torch.nn.Module, model_path: str = PATH_VAE, device: torch.device = None) -> torch.nn.Module:
    if device is None:
        device = get_device()
    if os.path.exists(model_path):
        state_dict = torch.load(
            model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        return model
    return None

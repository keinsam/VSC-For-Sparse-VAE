import torch
import torch.nn as nn
from logic.common import AE_LATENT_DIM


class Autoencoder(nn.Module):
    def __init__(self, latent_dim: int = AE_LATENT_DIM):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            # 32² -> 16²
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 16² -> 8²
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 8² -> 4²
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the autoencoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 32, 32).
                values between 0 and 1

        Returns:
            torch.Tensor: Reconstructed tensor of shape (batch_size, 1, 32, 32).
                values between 0 and 1
        """
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed

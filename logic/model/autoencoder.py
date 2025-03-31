import torch
import torch.nn as nn
from logic.common import AE_LATENT_DIM


class Autoencoder(nn.Module):
    def __init__(
            self,
            latent_dim: int = AE_LATENT_DIM,
            img_size: int = 28):
        super().__init__()
        self.img_size = img_size
        self.out1 = (img_size - 1) // 2 + 1
        self.out2 = (self.out1 - 1) // 2 + 1
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, self.out2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Unflatten(1, (64, 1, 1)),
            nn.ConvTranspose2d(64, 32, self.out2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 16, 3, 2, 1, output_padding=self.out1 - (2 * self.out2 - 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, 1, 3, 2, 1, output_padding=img_size - (2 * self.out1 - 1)),
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the autoencoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).
                values between 0 and 1

        Returns:
            torch.Tensor: Reconstructed tensor of shape (batch_size, 1, 28, 28).
                values between 0 and 1
        """
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed

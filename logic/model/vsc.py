import torch
import torch.nn as nn
from common import VSC_LATENT_DIM


class VSC(nn.Module):
    def __init__(self):
        super(VSC, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7),  # 7x7 -> 1x1
            nn.Flatten(),
        )

        self.linear_dim = 64
        self.fc_mu = nn.Linear(self.linear_dim, VSC_LATENT_DIM)
        self.fc_logvar = nn.Linear(self.linear_dim, VSC_LATENT_DIM)
        self.fc_decode = nn.Linear(VSC_LATENT_DIM, 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.linear_dim, 32, kernel_size=7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.Sigmoid()
        )
        self.sparsity_penalty = nn.L1Loss()

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x).view(-1, 64)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.fc_decode(z).view(-1, 64, 1, 1)
        x = self.decoder(z)
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the autoencoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 32, 32).
                values between 0 and 1

        Returns:
            torch.Tensor: Reconstructed tensor of shape (batch_size, 1, 32, 32).
                values between 0 and 1
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar

    def compute_sparsity_loss(self, z: torch.Tensor) -> torch.Tensor:
        return self.sparsity_penalty(z, torch.zeros_like(z))

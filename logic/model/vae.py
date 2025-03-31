import torch
import torch.nn as nn
from logic.common import VAE_LATENT_DIM


class VAE(nn.Module):
    def __init__(self, latent_dim: int = VAE_LATENT_DIM, img_size: int = 28):
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
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 64)
        self.decoder = nn.Sequential(
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
        x = self.encoder(x)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.fc_decode(z).view(-1, 64, 1, 1)
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

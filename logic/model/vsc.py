import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Tuple

# VSC model with explicit support prediction (gamma) for spike and slab.


class VSC(nn.Module):
    def __init__(self, latent_dim: int, img_size: int, prior_sparsity: float = 0.01) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.prior_sparsity = prior_sparsity
        # Lambda for spike warm-up; starts at 0 and increases to 1.
        self.lambda_val = 1.0

        # Compute spatial dimensions after two conv layers.
        self.out1 = (img_size + 2 * 1 - 3) // 2 + 1  # after first conv layer
        self.out2 = (self.out1 + 2 * 1 - 3) // 2 + \
            1   # after second conv layer

        # Encoder: three convolutional layers; last conv uses kernel size = out2 to yield a 1x1 feature map.
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2,
                      padding=1),  # img_size -> out1
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2,
                      padding=1),  # out1 -> out2
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=self.out2,
                      stride=1),      # out2 -> 1
            nn.Flatten()
        )
        # Fully connected layers for the continuous parameters.
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        # Additional layer to predict support (spike probability) for each latent dimension.
        self.fc_gamma = nn.Linear(64, latent_dim)

        # Decoder: mirror the encoder.
        self.fc_decode = nn.Linear(latent_dim, 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=self.out2, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1,
                output_padding=self.out1 - (2 * self.out2 - 1)
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, 1, kernel_size=3, stride=2, padding=1,
                output_padding=img_size - (2 * self.out1 - 1)
            ),
            nn.Sigmoid()
        )

    # Encoder forward pass returning continuous parameters and support.
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        gamma = torch.sigmoid(self.fc_gamma(features))
        return mu, logvar, gamma

    # Reparameterization with spike and slab.
    # For differentiability, one might use a continuous relaxation (e.g. Gumbel-Sigmoid);
    # here we use torch.bernoulli for simplicity.
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        lam = self.lambda_val  # warm-up factor
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        # Interpolate between a fixed (zero-mean, unit variance) slab and the learned slab.
        slab = lam * mu + eps * (lam * std + (1 - lam))
        # Sample binary spike; note: torch.bernoulli is not differentiable.
        spike = torch.bernoulli(gamma)
        return spike * slab

    # Decoder forward pass.
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z_fc = self.fc_decode(z)
        z_reshaped = z_fc.view(-1, 64, 1, 1)
        return self.decoder(z_reshaped)

    # Full forward pass returns reconstruction and latent parameters.
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar, gamma = self.encode(x)
        z = self.reparameterize(mu, logvar, gamma)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, gamma

    # Compute sparsity loss on gamma to push the support towards the prior sparsity level.
    def compute_sparsity_loss(self, gamma: torch.Tensor) -> torch.Tensor:
        target = torch.full_like(gamma, self.prior_sparsity)
        return nn.functional.binary_cross_entropy(gamma, target)

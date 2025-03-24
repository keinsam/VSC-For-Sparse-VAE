from matplotlib import pyplot as plt
import torch
from logic.common import VAE_LATENT_DIM


def visualize_latent_space(
        model: torch.nn.Module,
        latent_dim: int = VAE_LATENT_DIM,
        device: torch.device = None,
        text: str = "Latent Space Visualization") -> None:
    """Visualize the latent space by sampling a 2D grid."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    with torch.no_grad():
        # Create a 10x10 grid over two latent dimensions (-2 to 2)
        grid_x = torch.linspace(-2, 2, 10)
        grid_y = torch.linspace(-2, 2, 10)
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle(text, fontsize=16)

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                # Create latent vector with first two dims varying, others zero
                z = torch.zeros(1, latent_dim).to(device)
                z[0, 0] = xi
                z[0, 1] = yi
                # Decode to image
                x_decoded = model.decode(z)
                img = x_decoded[0, 0].cpu().numpy()  # Shape (28, 28)
                # Plot in grid
                plt.subplot(10, 10, i * 10 + j + 1)
                plt.imshow(img, cmap='gray')
                plt.axis('off')
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for title
        plt.show()

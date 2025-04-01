import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from torch.utils.data import DataLoader
from logic.common import PATH_VAE
from logic.model.base import load_model
from logic.model.vae import VAE
from logic.data import load_mnist, make_dataloader # TODO : to be changed for test set

def visualize_reconstruction(
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device = None,
        num_samples: int = 4,
        noise_level: float = 0.1,
        text: str = "Reconstruction Comparison") -> go.Figure:
    """Visualize the reconstruction of classic and noisy images."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get data samples
    x, _ = next(iter(dataloader))
    x = x[:num_samples].to(device)
    x_noisy = (x + noise_level * torch.randn_like(x)).clamp(0, 1)

    # Generate reconstructions
    model.eval()
    with torch.no_grad():
        # Check if model is VSC or VAE
        if hasattr(model, 'fc_gamma'):  # VSC case
            mu, logvar, gamma = model.encode(x)
            z = model.reparameterize(mu, logvar, gamma)
            mu_noisy, logvar_noisy, gamma_noisy = model.encode(x_noisy)
            z_noisy = model.reparameterize(mu_noisy, logvar_noisy, gamma_noisy)
        else:  # VAE case
            mu, logvar = model.encode(x)
            z = model.reparameterize(mu, logvar)
            mu_noisy, logvar_noisy = model.encode(x_noisy)
            z_noisy = model.reparameterize(mu_noisy, logvar_noisy)
        # Reconstruct images
        x_recon = model.decode(z)
        x_recon_noisy = model.decode(z_noisy)

    # Convert to numpy
    x = x.cpu().numpy()
    x_noisy = x_noisy.cpu().numpy()
    x_recon = x_recon.cpu().numpy()
    x_recon_noisy = x_recon_noisy.cpu().numpy()

    # Create figure
    fig = make_subplots(
        rows=num_samples,
        cols=4,
        subplot_titles=['Original', 'Noisy', 'Reconstruction', 'Noisy Reconstruction'],
        horizontal_spacing=0.02,
        vertical_spacing=0.05
    )

    # Add images
    for i in range(num_samples):
        for j, img in enumerate([x[i,0], x_noisy[i,0], x_recon[i,0], x_recon_noisy[i,0]]):
            fig.add_trace(
                go.Heatmap(z=img, colorscale='gray', showscale=False),
                row=i+1, col=j+1
            )
    fig.update_layout(
        title_text=text,
        height=200*num_samples,
        width=800,
        margin=dict(t=50)
    )
    # Hide x and y axes
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    fig.show()
    return fig


def main() -> None:
    model = load_model(VAE(), PATH_VAE)
    if model is None:
        print("Require a trained model")

    test_loader = make_dataloader(load_mnist()) # TODO : not a test loader, need to change dataloader functions

    visualize_reconstruction(model=model, dataloader=test_loader)


if __name__ == "__main__":
    main()
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from torch.utils.data import DataLoader
from logic.common import PATH_VAE
from logic.model.base import load_model
from logic.model.vae import VAE
from logic.data import load_mnist, make_dataloader


def forward_model(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        output = model(x)
        if isinstance(output, tuple):
            output = output[0]
    return output


def visualize_reconstruction(
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device = None,
        num_samples: int = 4,
        noise_level: float = 0.1,
        text: str = "Reconstruction Comparison") -> go.Figure:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, _ = next(iter(dataloader))
    x = x[:num_samples].to(device)
    x_noisy = (x + noise_level * torch.randn_like(x)).clamp(0, 1)
    x_recon = forward_model(model, x)
    x_recon_noisy = forward_model(model, x_noisy)
    x_np = x.cpu().numpy()
    x_noisy_np = x_noisy.cpu().numpy()
    x_recon_np = x_recon.cpu().numpy()
    x_recon_noisy_np = x_recon_noisy.cpu().numpy()
    fig = make_subplots(
        rows=num_samples,
        cols=4,
        subplot_titles=['Original', 'Noisy',
                        'Reconstruction', 'Noisy Reconstruction'],
        horizontal_spacing=0.02,
        vertical_spacing=0.05
    )
    for i in range(num_samples):
        for j, img in enumerate([x_np[i, 0], x_noisy_np[i, 0], x_recon_np[i, 0], x_recon_noisy_np[i, 0]]):
            fig.add_trace(
                go.Heatmap(z=img, colorscale='gray', showscale=False),
                row=i+1, col=j+1
            )
    fig.update_layout(
        title_text=text,
        height=200 * num_samples,
        width=800,
        margin=dict(t=50)
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


def main() -> None:
    model = load_model(VAE(), PATH_VAE)
    if model is None:
        print("Require a trained model")
        return
    test_loader = make_dataloader(load_mnist())
    fig = visualize_reconstruction(model=model, dataloader=test_loader)
    fig.show()


if __name__ == "__main__":
    main()

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from logic.common import VAE_LATENT_DIM, PATH_VAE
from logic.model.base import load_model
from logic.model.vae import VAE


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
        fig = make_subplots(
            rows=10, 
            cols=10,
            horizontal_spacing=0.01,
            vertical_spacing=0.01
        )

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                # Create latent vector with first two dims varying, others zero
                z = torch.zeros(1, latent_dim).to(device)
                z[0, 0] = xi
                z[0, 1] = yi
                # Decode to image
                x_decoded = model.decode(z)
                img = x_decoded[0, 0].cpu().numpy()
                # Plot in grid
                fig.add_trace(
                    go.Heatmap(
                        z=img,
                        colorscale='gray',
                        showscale=False,
                        hoverinfo='none'
                    ),
                    row=i+1, 
                    col=j+1
                )
        
        fig.update_layout(
            title_text=text,
            title_x=0.5,
            height=800,
            width=800,
            margin=dict(l=20, r=20, b=20, t=40),
        )

        # Hide x and y axes
        for i in range(1, 11):
            for j in range(1, 11):
                fig.update_xaxes(showticklabels=False, row=i, col=j)
                fig.update_yaxes(showticklabels=False, row=i, col=j)

        fig.show()
        return fig


def main() -> None:
    model = load_model(VAE(), PATH_VAE)
    if model is None:
        print("Require a trained model")
    visualize_latent_space(model, VAE_LATENT_DIM)


if __name__ == "__main__":
    main()
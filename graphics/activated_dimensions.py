import torch
import plotly.graph_objects as go
from torch.utils.data import DataLoader

def visualize_activated_dimensions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    num_categories: int = 10,
    device: torch.device = None,
    text: str = "Activated Dimensions per Category"
) -> go.Figure:
    """
    Visualize the mean latent representations for each category as a heatmap.

    Args:
        model (torch.nn.Module): The trained model (Autoencoder, VAE, or VSC).
        dataloader (DataLoader): DataLoader providing batches of images and labels.
        num_categories (int): Number of categories in the dataset (default: 10 for MNIST).
        device (torch.device): Device to run the model on (default: CUDA if available, else CPU).
        text (str): Title of the visualization.

    Returns:
        go.Figure: Plotly figure object containing the heatmap.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    
    # Determine latent dimension from the first batch
    x, _ = next(iter(dataloader))
    x = x.to(device)
    encode_output = model.encode(x)
    if isinstance(encode_output, tuple):
        z = encode_output[0]  # Use mu for VAE and VSC
    else:
        z = encode_output  # Use encoded vector for Autoencoder
    latent_dim = z.size(1)
    
    # Initialize tensors to accumulate sums and counts per category
    sum_z = torch.zeros(num_categories, latent_dim, device=device)
    count = torch.zeros(num_categories, device=device)
    
    # Compute mean latent representations
    with torch.no_grad():
        for x, labels in dataloader:
            x = x.to(device)
            labels = labels.to(device)
            encode_output = model.encode(x)
            if isinstance(encode_output, tuple):
                z = encode_output[0]  # mu for VAE and VSC
            else:
                z = encode_output  # encoded vector for Autoencoder
            for c in range(num_categories):
                mask = labels == c
                sum_z[c] += z[mask].sum(dim=0)
                count[c] += mask.sum()
    
    # Calculate mean, avoiding division by zero
    mean_z = sum_z / count.view(-1, 1).clamp(min=1e-6)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=mean_z.cpu().numpy(),
        x=[f'Dim {i}' for i in range(latent_dim)],
        y=[f'Category {c}' for c in range(num_categories)],
        colorscale='Viridis',
        colorbar=dict(title='Mean Activation')
    ))
    
    fig.update_layout(
        title_text=text,
        xaxis_title="Latent Dimensions",
        yaxis_title="Categories",
        height=600,
        width=800
    )
    
    return fig

def main() -> None:
    """Test the visualization with a VAE model on MNIST."""
    from logic.common import PATH_VAE
    from logic.model.vae import VAE
    from logic.data import load_mnist, make_dataloader
    from logic.model.base import load_model
    
    model = load_model(VAE(), PATH_VAE)
    if model is None:
        print("Require a trained model")
        return
    
    dataloader = make_dataloader(load_mnist())
    fig = visualize_activated_dimensions(model, dataloader)
    fig.show()

if __name__ == "__main__":
    main()
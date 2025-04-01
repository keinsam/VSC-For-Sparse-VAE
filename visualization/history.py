from typing import List, Dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_epochs(history: List[Dict]) -> List:
    return [entry["epoch"] for entry in history]


def get_loss_components(history: List[Dict]) -> List[str]:
    return sorted(
        [
            key
            for key in history[0].keys()
            if key.startswith("avg_") and key.endswith("_loss") and key != "avg_loss"
        ]
    )


def has_lambda(history: List[Dict]) -> bool:
    return "lambda" in history[0]


def add_loss_traces(fig: go.Figure, epochs: List, history: List[Dict]) -> None:
    components = get_loss_components(history)
    if components:
        for i, key in enumerate(components):
            fill = "tozeroy" if i == 0 else "tonexty"
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=[entry[key] for entry in history],
                    mode="lines",
                    name=key,
                    fill=fill,
                ),
                secondary_y=False,
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=[entry["avg_loss"] for entry in history],
                mode="lines",
                name="avg_loss",
            ),
            secondary_y=False,
        )


def add_lambda_trace(fig: go.Figure, epochs: List, history: List[Dict]) -> None:
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=[entry["lambda"] for entry in history],
            mode="lines+markers",
            name="lambda",
            line={"color": "black"},
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="lambda", secondary_y=True)


def visualize_history(history: List[Dict]) -> go.Figure:
    epochs = get_epochs(history)
    secondary = has_lambda(history)
    fig = (
        make_subplots(specs=[[{"secondary_y": True}]])
        if secondary
        else go.Figure()
    )
    add_loss_traces(fig, epochs, history)
    if secondary:
        add_lambda_trace(fig, epochs, history)
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss", secondary_y=False)
    fig.update_layout(title="Training History")
    return fig

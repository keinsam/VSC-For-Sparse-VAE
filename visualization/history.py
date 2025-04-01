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


def add_loss_traces(fig: go.Figure, epochs: List, history: List[Dict], use_subplots: bool) -> None:
    components = get_loss_components(history)
    if components:
        for i, key in enumerate(components):
            fill = "tozeroy" if i == 0 else "tonexty"
            trace = go.Scatter(
                x=epochs,
                y=[entry[key] for entry in history],
                mode="lines",
                name=key,
                fill=fill,
            )
            if use_subplots:
                fig.add_trace(trace, secondary_y=False)
            else:
                fig.add_trace(trace)
    else:
        trace = go.Scatter(
            x=epochs,
            y=[entry["avg_loss"] for entry in history],
            mode="lines",
            name="avg_loss",
        )
        if use_subplots:
            fig.add_trace(trace, secondary_y=False)
        else:
            fig.add_trace(trace)


def add_lambda_trace(fig: go.Figure, epochs: List, history: List[Dict]) -> None:
    trace = go.Scatter(
        x=epochs,
        y=[entry["lambda"] for entry in history],
        mode="lines+markers",
        name="lambda",
        line={"color": "black"},
    )
    fig.add_trace(trace, secondary_y=True)
    fig.update_yaxes(title_text="lambda", secondary_y=True)


def visualize_history(history: List[Dict]) -> go.Figure:
    epochs = get_epochs(history)
    use_subplots = has_lambda(history)
    if use_subplots:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    else:
        fig = go.Figure()
    add_loss_traces(fig, epochs, history, use_subplots)
    if use_subplots:
        add_lambda_trace(fig, epochs, history)
        fig.update_yaxes(title_text="Loss", secondary_y=False)
    else:
        fig.update_yaxes(title_text="Loss")
    fig.update_xaxes(title_text="Epoch")
    fig.update_layout(title="Training History")
    return fig

import plotly.graph_objects as go
import numpy as np

def plot_signal(signal: np.ndarray, fs: int, title: str = "Segnale", peaks: list = None, color: str = "#1f77b4") -> go.Figure:
    """Grafico interattivo del segnale con annotazioni picchi opzionali."""
    time = np.arange(len(signal)) / fs
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time, y=signal, mode="lines", name=title,
        line=dict(color=color, width=1),
    ))
    if peaks is not None and len(peaks) > 0:
        peaks = np.array(peaks)
        valid = peaks[peaks < len(signal)]
        fig.add_trace(go.Scatter(
            x=time[valid], y=signal[valid],
            mode="markers", name="Picchi",
            marker=dict(color="red", size=6, symbol="triangle-up"),
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Tempo (s)",
        yaxis_title="Ampiezza",
        height=350,
        margin=dict(l=50, r=20, t=40, b=40),
        hovermode="x unified",
    )
    return fig

def plot_signal_comparison(raw: np.ndarray, clean: np.ndarray, fs: int) -> go.Figure:
    """Confronto segnale grezzo vs preprocessato."""
    time = np.arange(len(raw)) / fs
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=raw, mode="lines", name="Grezzo", line=dict(color="gray", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=time, y=clean, mode="lines", name="Preprocessato", line=dict(color="#2ecc71", width=1.5)))
    fig.update_layout(title="Raw vs Preprocessato", xaxis_title="Tempo (s)", yaxis_title="Ampiezza", height=350)
    return fig

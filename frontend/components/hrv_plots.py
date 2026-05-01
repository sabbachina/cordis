import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_poincare(rr_intervals: list[float]) -> go.Figure:
    """Scatter plot Poincaré RR(n) vs RR(n+1)."""
    rr = np.array(rr_intervals)
    if len(rr) < 4:
        fig = go.Figure()
        fig.add_annotation(text="Dati insufficienti", x=0.5, y=0.5, showarrow=False)
        return fig
    x, y = rr[:-1], rr[1:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(color="#3498db", size=5, opacity=0.6), name="RR pairs"))
    # SD1/SD2 ellipse approximation
    sd1 = np.std(y - x) / np.sqrt(2)
    sd2 = np.std(y + x) / np.sqrt(2)
    cx, cy = np.mean(x), np.mean(y)
    fig.add_annotation(x=cx, y=cy, text=f"SD1={sd1:.1f}ms<br>SD2={sd2:.1f}ms", showarrow=False, bgcolor="white", bordercolor="#3498db")
    fig.update_layout(title="Poincaré Plot HRV", xaxis_title="RR(n) ms", yaxis_title="RR(n+1) ms", height=350)
    return fig

def plot_hrv_spectrum(vlf: float, lf: float, hf: float) -> go.Figure:
    """Bar chart potenza spettrale HRV per bande di frequenza."""
    fig = go.Figure()
    categories = ["VLF (0–0.04 Hz)", "LF (0.04–0.15 Hz)", "HF (0.15–0.4 Hz)"]
    values = [vlf or 0, lf or 0, hf or 0]
    colors = ["#9b59b6", "#3498db", "#2ecc71"]
    fig.add_trace(go.Bar(x=categories, y=values, marker_color=colors, text=[f"{v:.1f} ms²" for v in values], textposition="auto"))
    fig.update_layout(title="Spettro HRV per Bande di Frequenza", yaxis_title="Potenza (ms²)", height=300)
    return fig

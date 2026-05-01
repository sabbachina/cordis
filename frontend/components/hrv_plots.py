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


def plot_rr_tachogram(rr_intervals_ms: list) -> go.Figure:
    """RR interval tachogram — variazioni nel tempo."""
    rr = np.array(rr_intervals_ms)
    if len(rr) < 3:
        fig = go.Figure()
        fig.add_annotation(text="Dati insufficienti", x=0.5, y=0.5, showarrow=False)
        return fig
    beat_n = np.arange(len(rr))

    # Linea di tendenza
    poly = np.polyfit(beat_n, rr, 1)
    trend = np.polyval(poly, beat_n)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=beat_n.tolist(), y=rr.tolist(),
        mode="lines+markers",
        name="RR interval",
        line=dict(color="#2980b9", width=1.5),
        marker=dict(size=3),
    ))
    fig.add_trace(go.Scatter(
        x=beat_n.tolist(), y=trend.tolist(),
        mode="lines", name="Trend",
        line=dict(color="#e74c3c", width=1, dash="dash"),
    ))
    # Zone normali
    fig.add_hrect(y0=600, y1=1000, fillcolor="rgba(46,204,113,0.05)", line_width=0,
                  annotation_text="Normal zone")
    fig.update_layout(
        title="Tachogramma RR (intervalli battito-battito)",
        xaxis_title="Numero battito",
        yaxis_title="RR interval (ms)",
        height=300,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


def plot_rr_histogram(rr_intervals_ms: list) -> go.Figure:
    """Istogramma distribuzione RR con curva normale sovrapposta."""
    from scipy import stats as scipy_stats
    rr = np.array(rr_intervals_ms)
    if len(rr) < 5:
        fig = go.Figure()
        fig.add_annotation(text="Dati insufficienti", x=0.5, y=0.5, showarrow=False)
        return fig

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=rr.tolist(), nbinsx=30,
        name="Distribuzione RR",
        marker_color="#3498db",
        opacity=0.7,
        histnorm="probability density",
    ))

    # Curva normale
    mu, sigma = float(np.mean(rr)), float(np.std(rr))
    x_range = np.linspace(rr.min(), rr.max(), 200)
    pdf = scipy_stats.norm.pdf(x_range, mu, sigma)
    fig.add_trace(go.Scatter(
        x=x_range.tolist(), y=pdf.tolist(),
        mode="lines", name=f"N(mu={mu:.0f}, sigma={sigma:.0f})",
        line=dict(color="#e74c3c", width=2),
    ))
    fig.update_layout(
        title=f"Distribuzione RR  [mu={mu:.0f}ms, sigma={sigma:.0f}ms]",
        xaxis_title="RR interval (ms)",
        yaxis_title="Densita",
        height=280,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


def plot_stft_heatmap(stft_data: dict) -> go.Figure:
    """Heatmap STFT power over time — Kubios-style time-frequency spectrogram."""
    if not stft_data or not stft_data.get("times"):
        fig = go.Figure()
        fig.add_annotation(
            text="Dati STFT non disponibili (richiede >=120s di segnale)",
            x=0.5, y=0.5, showarrow=False,
        )
        return fig

    times = stft_data["times"]
    freqs = stft_data["freqs"]
    power_db = stft_data["power_db"]

    fig = go.Figure(data=go.Heatmap(
        z=power_db,
        x=times,
        y=freqs,
        colorscale="Viridis",
        colorbar=dict(title="Power (dB)"),
        zmin=-20, zmax=20,
    ))

    # Add HRV band markers
    for f, label, color in [
        (0.04, "VLF/LF", "rgba(155,89,182,0.4)"),
        (0.15, "LF/HF", "rgba(52,152,219,0.4)"),
        (0.40, "HF limit", "rgba(46,204,113,0.4)"),
    ]:
        if freqs and f <= max(freqs):
            fig.add_hline(
                y=f, line=dict(color=color, width=1, dash="dash"),
                annotation_text=f"{label} ({f}Hz)",
            )

    fig.update_layout(
        title="STFT Time-Frequency Spectrogram (HRV)",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        height=320,
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


def plot_lf_hf_over_time(stft_data: dict) -> go.Figure:
    """LF and HF power over time from STFT analysis."""
    if not stft_data:
        fig = go.Figure()
        fig.add_annotation(text="N/D", x=0.5, y=0.5, showarrow=False)
        return fig

    times = stft_data.get("times", [])
    lf = stft_data.get("lf_over_time", [])
    hf = stft_data.get("hf_over_time", [])

    fig = go.Figure()
    if lf:
        fig.add_trace(go.Scatter(
            x=times, y=lf, mode="lines",
            name="LF Power", line=dict(color="#3498db", width=2),
        ))
    if hf:
        fig.add_trace(go.Scatter(
            x=times, y=hf, mode="lines",
            name="HF Power", line=dict(color="#2ecc71", width=2),
        ))
    fig.update_layout(
        title="LF / HF Power Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Power (ms^2)",
        height=260,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


def plot_hrv_psd(rr_intervals_ms: list) -> go.Figure:
    """Power Spectral Density continuo (Welch) con bande colorate."""
    from scipy import signal as scipy_signal
    from scipy.interpolate import interp1d

    rr = np.array(rr_intervals_ms)
    if len(rr) < 10:
        fig = go.Figure()
        fig.add_annotation(text="Dati insufficienti (min 10 RR)", x=0.5, y=0.5, showarrow=False)
        return fig

    fs_resample = 4.0
    t_rr = np.cumsum(rr) / 1000.0
    t_rr = t_rr - t_rr[0]
    t_uniform = np.linspace(t_rr[0], t_rr[-1], int(t_rr[-1] * fs_resample))

    if len(t_uniform) < 8:
        fig = go.Figure()
        fig.add_annotation(text="Segnale troppo breve per PSD", x=0.5, y=0.5, showarrow=False)
        return fig

    try:
        interp = interp1d(t_rr, rr, kind="cubic", bounds_error=False, fill_value="extrapolate")
        rr_uniform = interp(t_uniform)
        nperseg = min(len(rr_uniform), max(64, len(rr_uniform) // 4))
        freqs, psd = scipy_signal.welch(rr_uniform, fs=fs_resample, nperseg=nperseg)
    except Exception:
        fig = go.Figure()
        fig.add_annotation(text="Errore calcolo PSD", x=0.5, y=0.5, showarrow=False)
        return fig

    fig = go.Figure()
    # Aree colorate per bande
    bands = [
        (0.003, 0.04, "rgba(155,89,182,0.15)", "VLF"),
        (0.04, 0.15, "rgba(52,152,219,0.20)", "LF"),
        (0.15, 0.40, "rgba(46,204,113,0.20)", "HF"),
    ]
    for f_low, f_high, color, name in bands:
        mask = (freqs >= f_low) & (freqs <= f_high)
        if mask.any():
            fig.add_trace(go.Scatter(
                x=freqs[mask].tolist(), y=psd[mask].tolist(),
                fill="tozeroy", fillcolor=color,
                line=dict(width=0),
                name=name, mode="lines",
            ))
    # Curva PSD
    mask_full = freqs <= 0.40
    fig.add_trace(go.Scatter(
        x=freqs[mask_full].tolist(), y=psd[mask_full].tolist(),
        mode="lines", name="PSD",
        line=dict(color="#2c3e50", width=1.5),
    ))
    fig.update_layout(
        title="Power Spectral Density (Welch) — Bande HRV",
        xaxis_title="Frequenza (Hz)",
        yaxis_title="PSD (ms^2/Hz)",
        height=300,
        margin=dict(l=50, r=20, t=40, b=40),
        xaxis=dict(range=[0, 0.40]),
    )
    return fig

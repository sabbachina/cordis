import streamlit as st
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.signal_plot import plot_signal
from components.hrv_plots import plot_poincare

st.set_page_config(page_title="Step 3 — Peak Detection", page_icon="📍", layout="wide")
st.title("📍 Step 3: Rilevamento Picchi")

if st.session_state.get("signal_data") is None:
    st.warning("Vai prima allo Step 1.")
    st.stop()

sig = st.session_state.get("clean_signal") or st.session_state.signal_data
fs = st.session_state.sampling_rate
signal_type = st.session_state.signal_type

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Algoritmo di Rilevamento")

    if signal_type == "ECG":
        algorithm = st.selectbox("Algoritmo QRS", ["neurokit", "pantompkins1985", "hamilton2002", "engzeemod2012"],
                                help="Pan-Tompkins: standard gold. NeuroKit: automatico adattivo.")
        st.markdown("**Picchi cercati:** Complessi QRS (onde R)")
    else:
        algorithm = st.selectbox("Algoritmo PPG", ["elgendi", "bishop"],
                                help="Elgendi: robusto per PPG da wearable.")
        st.markdown("**Picchi cercati:** Picchi sistolici PPG")

    min_hr = st.slider("HR minima attesa (bpm)", 30, 80, 40)
    max_hr = st.slider("HR massima attesa (bpm)", 80, 250, 200)

    if st.button("🔍 Rileva Picchi", type="primary"):
        with st.spinner("Rilevamento in corso..."):
            try:
                import neurokit2 as nk
                if signal_type == "ECG":
                    _, info = nk.ecg_peaks(sig, sampling_rate=fs, method=algorithm)
                    peaks = info["ECG_R_Peaks"]
                else:
                    _, info = nk.ppg_peaks(sig, sampling_rate=fs, method=algorithm)
                    peaks = info["PPG_Peaks"]

                if len(peaks) < 3:
                    st.error(f"Solo {len(peaks)} picchi trovati. Controlla il segnale o cambia algoritmo.")
                else:
                    st.session_state.peaks = peaks
                    rr = np.diff(peaks) / fs * 1000
                    mean_hr = 60000 / np.mean(rr)
                    st.success(f"✅ {len(peaks)} picchi rilevati | HR media: {mean_hr:.1f} bpm")
            except Exception as e:
                st.error(f"Errore: {e}")

with col2:
    st.subheader("Segnale con Picchi")
    peaks = st.session_state.get("peaks")
    if peaks is not None and len(peaks) > 0:
        preview_len = min(len(sig), fs * 10)
        # Filtra picchi nel range di preview
        peaks_preview = peaks[peaks < preview_len]
        fig = plot_signal(sig[:preview_len], fs,
                         title=f"{signal_type} — {len(peaks)} picchi rilevati",
                         peaks=peaks_preview,
                         color="#e74c3c" if signal_type == "ECG" else "#3498db")
        st.plotly_chart(fig, use_container_width=True)

        # Statistiche RR
        rr_ms = np.diff(peaks) / fs * 1000
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("N. picchi", len(peaks))
        col_b.metric("HR media", f"{60000/np.mean(rr_ms):.1f} bpm")
        col_c.metric("RR medio", f"{np.mean(rr_ms):.1f} ms")
        col_d.metric("RR std", f"{np.std(rr_ms):.1f} ms")

        # Poincaré preview
        if len(rr_ms) >= 10:
            st.subheader("Poincaré Plot (preview)")
            fig_p = plot_poincare(rr_ms.tolist())
            st.plotly_chart(fig_p, use_container_width=True)

        st.success("✅ Vai al **Step 4: Biomarker** →")
    else:
        preview_len = min(len(sig), fs * 10)
        fig = plot_signal(sig[:preview_len], fs, title=f"{signal_type} — Nessun picco rilevato ancora")
        st.plotly_chart(fig, use_container_width=True)
        st.info("Clicca 'Rileva Picchi' per procedere.")

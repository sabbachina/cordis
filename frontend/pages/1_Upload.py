import streamlit as st
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sample_data import generate_sample_ecg, generate_sample_ppg
from utils.api_client import check_backend_health
from components.signal_plot import plot_signal

st.set_page_config(page_title="Step 1 — Upload", page_icon="📤", layout="wide")
st.title("📤 Step 1: Upload Segnale")
st.markdown("Carica il tuo file ECG o PPG, oppure usa i segnali campione per esplorare la piattaforma.")

# Backend status
backend_ok = check_backend_health()
if backend_ok:
    st.success("✅ Backend connesso")
else:
    st.warning("⚠️ Backend non raggiungibile — modalità offline (solo dati campione)")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Configurazione Segnale")
    signal_type = st.selectbox("Tipo di segnale", ["ECG", "PPG"], help="ECG = Elettrocardiogramma | PPG = Fotopletismografia")

    source = st.radio("Sorgente dati", ["Dati campione (demo)", "Carica file"])

    if source == "Dati campione (demo)":
        fs_default = 500 if signal_type == "ECG" else 125
        st.info(f"Verrà generato un segnale {signal_type} sintetico di 30 secondi a {fs_default} Hz")
        if st.button("🔄 Genera segnale campione", type="primary"):
            with st.spinner("Generazione in corso..."):
                if signal_type == "ECG":
                    sig, fs = generate_sample_ecg()
                else:
                    sig, fs = generate_sample_ppg()
            st.session_state.signal_data = sig
            st.session_state.signal_type = signal_type
            st.session_state.sampling_rate = fs
            st.session_state.clean_signal = None
            st.session_state.peaks = None
            st.session_state.biomarker_report = None
            st.success(f"Segnale {signal_type} generato: {len(sig)} campioni a {fs} Hz")
    else:
        uploaded = st.file_uploader("Carica file", type=["csv", "txt", "xlsx", "xls", "edf", "json"])
        fs_manual = st.number_input("Frequenza di campionamento (Hz)", min_value=50, max_value=2000, value=500 if signal_type == "ECG" else 125)
        col_name = st.text_input("Nome colonna segnale (CSV/Excel)", value="amplitude", help="Lascia vuoto per usare la prima colonna numerica")

        if uploaded and st.button("📂 Carica segnale", type="primary"):
            with st.spinner("Caricamento..."):
                try:
                    suffix = os.path.splitext(uploaded.name)[1].lower()
                    if suffix in (".csv", ".txt"):
                        df = pd.read_csv(uploaded)
                        col = col_name if col_name and col_name in df.columns else df.select_dtypes(include=[np.number]).columns[0]
                        sig = df[col].dropna().values.astype(float)
                        fs = fs_manual
                    elif suffix in (".xlsx", ".xls"):
                        df = pd.read_excel(uploaded)
                        col = col_name if col_name and col_name in df.columns else df.select_dtypes(include=[np.number]).columns[0]
                        sig = df[col].dropna().values.astype(float)
                        fs = fs_manual
                    elif suffix == ".json":
                        import json
                        data = json.load(uploaded)
                        sig = np.array(data.get("values", data.get("signal", [])))
                        fs = data.get("sampling_rate", fs_manual)
                    else:
                        st.error(f"Formato {suffix} richiede il backend. Avvia docker-compose.")
                        sig, fs = None, None

                    if sig is not None and len(sig) > 0:
                        st.session_state.signal_data = sig
                        st.session_state.signal_type = signal_type
                        st.session_state.sampling_rate = fs
                        st.session_state.clean_signal = None
                        st.session_state.peaks = None
                        st.session_state.biomarker_report = None
                        st.success(f"Caricati {len(sig)} campioni a {fs} Hz")
                except Exception as e:
                    st.error(f"Errore caricamento: {e}")

with col2:
    st.subheader("Anteprima Segnale")
    if st.session_state.signal_data is not None:
        sig = st.session_state.signal_data
        fs = st.session_state.sampling_rate
        # Mostra solo i primi 10 secondi per performance
        preview_len = min(len(sig), fs * 10)
        fig = plot_signal(sig[:preview_len], fs, title=f"{st.session_state.signal_type} — Anteprima (primi 10s)")
        st.plotly_chart(fig, use_container_width=True)

        info_cols = st.columns(4)
        info_cols[0].metric("Campioni", f"{len(sig):,}")
        info_cols[1].metric("Frequenza", f"{fs} Hz")
        info_cols[2].metric("Durata", f"{len(sig)/fs:.1f} s")
        info_cols[3].metric("Tipo", st.session_state.signal_type)

        st.success("✅ Segnale caricato! Vai al **Step 2: Preprocessing** →")
    else:
        st.info("Nessun segnale caricato. Usa il pannello a sinistra.")

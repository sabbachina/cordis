import streamlit as st
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.signal_plot import plot_signal, plot_signal_comparison

st.set_page_config(page_title="Step 2 — Preprocessing", page_icon="🔧", layout="wide")
st.title("🔧 Step 2: Preprocessing del Segnale")

if st.session_state.get("signal_data") is None:
    st.warning("⚠️ Nessun segnale caricato. Vai prima allo Step 1.")
    st.stop()

sig = st.session_state.signal_data
fs = st.session_state.sampling_rate
signal_type = st.session_state.signal_type

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Parametri Filtro")

    if signal_type == "ECG":
        lowcut = st.slider("Frequenza di taglio bassa (Hz)", 0.1, 5.0, 0.5, 0.1,
                          help="Rimuove baseline wander. Valore tipico ECG: 0.5 Hz")
        highcut = st.slider("Frequenza di taglio alta (Hz)", 5.0, 150.0, 40.0, 1.0,
                           help="Rimuove rumore ad alta frequenza. Tipico ECG: 40 Hz")
    else:
        lowcut = st.slider("Frequenza di taglio bassa (Hz)", 0.1, 2.0, 0.5, 0.1,
                          help="Rimuove deriva lenta. Tipico PPG: 0.5 Hz")
        highcut = st.slider("Frequenza di taglio alta (Hz)", 2.0, 30.0, 8.0, 0.5,
                           help="Rimuove rumore. Tipico PPG: 8 Hz")

    method = st.selectbox("Metodo filtro", ["butterworth", "savgol", "neurokit"],
                         help="Butterworth: classico. Savitzky-Golay: preserva picchi. NeuroKit: automatico.")
    remove_baseline = st.checkbox("Rimuovi baseline wander", value=True)

    st.markdown("---")
    st.markdown("**Frequenze diagnostiche:**")
    if signal_type == "ECG":
        st.markdown("- Componente QRS: 5–40 Hz\n- Onda P/T: 0.5–10 Hz\n- Baseline wander: < 0.5 Hz")
    else:
        st.markdown("- PPG fondamentale: ~1–2 Hz (60–120 bpm)\n- Armoniche: fino a 8 Hz\n- Baseline: < 0.5 Hz")

    if st.button("⚙️ Applica Preprocessing", type="primary"):
        with st.spinner("Filtraggio in corso..."):
            try:
                from scipy.signal import butter, sosfilt, iirnotch, filtfilt
                from scipy.signal import savgol_filter

                nyquist = fs / 2
                lc = max(0.01, min(lowcut, nyquist * 0.9))
                hc = max(lc + 0.1, min(highcut, nyquist * 0.99))

                # Butterworth bandpass
                sos = butter(4, [lc / nyquist, hc / nyquist], btype='band', output='sos')
                clean = sosfilt(sos, sig)

                # Notch 50Hz
                if 50 < nyquist:
                    b, a = iirnotch(50, 30, fs)
                    clean = filtfilt(b, a, clean)

                st.session_state.clean_signal = clean
                st.session_state.preprocessing_config = {
                    "lowcut": lowcut, "highcut": highcut,
                    "method": method, "remove_baseline": remove_baseline
                }
                st.success("✅ Preprocessing completato!")
            except Exception as e:
                st.error(f"Errore preprocessing: {e}")

with col2:
    st.subheader("Confronto: Raw vs Preprocessato")
    clean = st.session_state.get("clean_signal")
    if clean is not None:
        preview_len = min(len(sig), fs * 10)
        fig = plot_signal_comparison(sig[:preview_len], clean[:preview_len], fs)
        st.plotly_chart(fig, use_container_width=True)

        noise_reduction = 1 - (np.std(clean) / np.std(sig))
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Riduzione rumore", f"{noise_reduction*100:.1f}%")
        col_b.metric("Ampiezza raw", f"{np.std(sig):.4f}")
        col_c.metric("Ampiezza clean", f"{np.std(clean):.4f}")
        st.success("✅ Vai al **Step 3: Peak Detection** →")
    else:
        preview_len = min(len(sig), fs * 10)
        fig = plot_signal(sig[:preview_len], fs, title=f"Segnale Raw ({signal_type})")
        st.plotly_chart(fig, use_container_width=True)
        st.info("Configura i parametri e clicca 'Applica Preprocessing'")

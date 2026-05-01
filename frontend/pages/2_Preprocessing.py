import streamlit as st
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.i18n import t, render_language_selector
from components.signal_plot import plot_signal, plot_signal_comparison

st.set_page_config(page_title="Step 2 — Preprocessing", page_icon="", layout="wide")

render_language_selector()

st.title(t("step2_title"))

if st.session_state.get("signal_data") is None:
    st.warning(t("go_step1"))
    st.stop()

sig = st.session_state.signal_data
fs = st.session_state.sampling_rate
signal_type = st.session_state.signal_type

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader(t("filter_params"))

    if signal_type == "ECG":
        lowcut = st.slider(t("lowcut_label"), 0.1, 5.0, 0.5, 0.1, help=t("lowcut_ecg_help"))
        highcut = st.slider(t("highcut_label"), 5.0, 150.0, 40.0, 1.0, help=t("highcut_ecg_help"))
    else:
        lowcut = st.slider(t("lowcut_label"), 0.1, 2.0, 0.5, 0.1, help=t("lowcut_ppg_help"))
        highcut = st.slider(t("highcut_label"), 2.0, 30.0, 8.0, 0.5, help=t("highcut_ppg_help"))

    method = st.selectbox(t("method_label"), ["butterworth", "savgol", "neurokit"], help=t("method_help"))
    remove_baseline = st.checkbox(t("remove_baseline"), value=True)

    st.markdown("---")
    st.markdown(t("diag_freqs"))
    if signal_type == "ECG":
        st.markdown(t("ecg_freq_info"))
    else:
        st.markdown(t("ppg_freq_info"))

    if st.button(t("btn_sqi")):
        with st.spinner(t("spinner_sqi")):
            try:
                _backend_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    '..', 'backend'
                )
                sys.path.insert(0, _backend_dir)
                from core.signal_quality import SignalQualityAnalyzer
                sqi_result = SignalQualityAnalyzer.compute_sqi(sig, fs, signal_type)
                st.session_state.sqi_result = sqi_result
                st.success(t("sqi_ok"))
            except Exception as e:
                st.error(t("sqi_err", err=e))

    if st.button(t("btn_preprocess"), type="primary"):
        with st.spinner(t("spinner_preprocess")):
            try:
                from scipy.signal import butter, sosfilt, iirnotch, filtfilt

                nyquist = fs / 2
                lc = max(0.01, min(lowcut, nyquist * 0.9))
                hc = max(lc + 0.1, min(highcut, nyquist * 0.99))

                sos = butter(4, [lc / nyquist, hc / nyquist], btype='band', output='sos')
                clean = sosfilt(sos, sig)

                if 50 < nyquist:
                    b, a = iirnotch(50, 30, fs)
                    clean = filtfilt(b, a, clean)

                st.session_state.clean_signal = clean
                st.session_state.preprocessing_config = {
                    "lowcut": lowcut, "highcut": highcut,
                    "method": method, "remove_baseline": remove_baseline
                }
                st.success(t("preprocess_ok"))
            except Exception as e:
                st.error(t("preprocess_err", err=e))

with col2:
    st.subheader(t("compare_title"))

    sqi_result = st.session_state.get("sqi_result")
    if sqi_result is not None:
        from components.biomarker_panel import render_sqi_widget
        render_sqi_widget(sqi_result)
        if sqi_result.get("overall_score", 100) < 40:
            st.error(t("sqi_poor"))

    clean = st.session_state.get("clean_signal")
    if clean is not None:
        preview_len = min(len(sig), fs * 10)
        fig = plot_signal_comparison(sig[:preview_len], clean[:preview_len], fs)
        st.plotly_chart(fig, use_container_width=True)

        noise_reduction = 1 - (np.std(clean) / np.std(sig))
        col_a, col_b, col_c = st.columns(3)
        col_a.metric(t("noise_reduction"), f"{noise_reduction*100:.1f}%")
        col_b.metric(t("amplitude_raw"), f"{np.std(sig):.4f}")
        col_c.metric(t("amplitude_clean"), f"{np.std(clean):.4f}")
        st.success(t("step2_done"))
    else:
        preview_len = min(len(sig), fs * 10)
        fig = plot_signal(sig[:preview_len], fs, title=f"Raw ({signal_type})")
        st.plotly_chart(fig, use_container_width=True)
        st.info(t("preprocess_hint"))

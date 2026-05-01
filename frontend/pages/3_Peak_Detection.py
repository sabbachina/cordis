import streamlit as st
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.i18n import t, render_language_selector
from components.signal_plot import plot_signal
from components.hrv_plots import plot_poincare, plot_rr_tachogram, plot_rr_histogram

st.set_page_config(page_title="Step 3 — Peak Detection", page_icon="", layout="wide")

render_language_selector()

st.title(t("step3_title"))

if st.session_state.get("signal_data") is None:
    st.warning(t("go_step1_warn"))
    st.stop()

_clean = st.session_state.get("clean_signal")
sig = _clean if _clean is not None else st.session_state.signal_data
fs = st.session_state.sampling_rate
signal_type = st.session_state.signal_type

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader(t("peak_algorithm"))

    if signal_type == "ECG":
        algorithm = st.selectbox(
            t("qrs_algo_label"),
            ["neurokit", "pantompkins1985", "hamilton2002", "engzeemod2012"],
            help=t("qrs_algo_help"),
        )
        st.markdown(t("ecg_peak_info"))
    else:
        algorithm = st.selectbox(
            t("ppg_algo_label"),
            ["elgendi", "bishop"],
            help=t("ppg_algo_help"),
        )
        st.markdown(t("ppg_peak_info"))

    min_hr = st.slider(t("min_hr_label"), 30, 80, 40)
    max_hr = st.slider(t("max_hr_label"), 80, 250, 200)

    if st.button(t("btn_detect"), type="primary"):
        with st.spinner(t("spinner_detect")):
            try:
                import neurokit2 as nk
                if signal_type == "ECG":
                    _, info = nk.ecg_peaks(sig, sampling_rate=fs, method=algorithm)
                    peaks = info["ECG_R_Peaks"]
                else:
                    _, info = nk.ppg_peaks(sig, sampling_rate=fs, method=algorithm)
                    peaks = info["PPG_Peaks"]

                if len(peaks) < 3:
                    st.error(t("too_few_peaks", n=len(peaks)))
                else:
                    st.session_state.peaks = peaks
                    rr = np.diff(peaks) / fs * 1000
                    mean_hr = 60000 / np.mean(rr)
                    st.success(t("peaks_ok", n=len(peaks), hr=mean_hr))
            except Exception as e:
                st.error(t("peak_err", err=e))

with col2:
    st.subheader(t("signal_with_peaks"))
    peaks = st.session_state.get("peaks")
    if peaks is not None and len(peaks) > 0:
        preview_len = min(len(sig), fs * 10)
        peaks_preview = peaks[peaks < preview_len]
        fig = plot_signal(
            sig[:preview_len], fs,
            title=t("plot_peaks_title", st=signal_type, n=len(peaks)),
            peaks=peaks_preview,
            color="#e74c3c" if signal_type == "ECG" else "#3498db",
        )
        st.plotly_chart(fig, use_container_width=True)

        rr_ms = np.diff(peaks) / fs * 1000
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric(t("metric_n_peaks"), len(peaks))
        col_b.metric(t("metric_hr_mean"), f"{60000/np.mean(rr_ms):.1f} bpm")
        col_c.metric(t("metric_rr_mean"), f"{np.mean(rr_ms):.1f} ms")
        col_d.metric(t("metric_rr_std"), f"{np.std(rr_ms):.1f} ms")

        if len(rr_ms) >= 10:
            tab1, tab2, tab3 = st.tabs([t("tab_tachogram"), t("tab_rr_dist"), t("tab_poincare")])
            with tab1:
                st.plotly_chart(plot_rr_tachogram(rr_ms.tolist()), use_container_width=True)
            with tab2:
                st.plotly_chart(plot_rr_histogram(rr_ms.tolist()), use_container_width=True)
            with tab3:
                st.plotly_chart(plot_poincare(rr_ms.tolist()), use_container_width=True)

        st.success(t("step3_done"))
    else:
        preview_len = min(len(sig), fs * 10)
        fig = plot_signal(sig[:preview_len], fs, title=t("plot_no_peaks", st=signal_type))
        st.plotly_chart(fig, use_container_width=True)
        st.info(t("detect_hint"))

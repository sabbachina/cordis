import streamlit as st
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.i18n import t, render_language_selector
from utils.sample_data import generate_sample_ecg, generate_sample_ppg
from utils.api_client import check_backend_health
from components.signal_plot import plot_signal

st.set_page_config(page_title="Step 1 — Upload", page_icon="", layout="wide")

render_language_selector()

st.title(t("step1_title"))
st.markdown(t("step1_subtitle"))

backend_ok = check_backend_health()
if backend_ok:
    st.success(t("backend_ok"))
else:
    st.warning(t("backend_fail"))

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader(t("signal_config"))
    signal_type = st.selectbox(
        t("signal_type_label"),
        ["ECG", "PPG"],
        help=t("signal_type_help"),
    )

    source = st.radio(
        t("data_source"),
        [t("source_sample"), t("source_upload")],
    )

    if source == t("source_sample"):
        fs_default = 500 if signal_type == "ECG" else 125
        st.info(t("sample_info", st=signal_type, fs=fs_default))
        if st.button(t("btn_generate"), type="primary"):
            with st.spinner(t("spinner_generate")):
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
            st.success(t("generated_ok", st=signal_type, n=len(sig), fs=fs))
    else:
        uploaded = st.file_uploader(t("uploader_label"), type=["csv", "txt", "xlsx", "xls", "edf", "json"])
        fs_manual = st.number_input(t("fs_label"), min_value=50, max_value=2000,
                                    value=500 if signal_type == "ECG" else 125)
        col_name = st.text_input(t("col_name_label"), value="amplitude", help=t("col_name_help"))

        if uploaded and st.button(t("btn_upload"), type="primary"):
            with st.spinner(t("spinner_upload")):
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
                        st.error(t("upload_format_err", ext=suffix))
                        sig, fs = None, None

                    if sig is not None and len(sig) > 0:
                        st.session_state.signal_data = sig
                        st.session_state.signal_type = signal_type
                        st.session_state.sampling_rate = fs
                        st.session_state.clean_signal = None
                        st.session_state.peaks = None
                        st.session_state.biomarker_report = None
                        st.success(t("upload_ok", n=len(sig), fs=fs))
                except Exception as e:
                    st.error(t("upload_err", err=e))

with col2:
    st.subheader(t("preview_title"))
    if st.session_state.signal_data is not None:
        sig = st.session_state.signal_data
        fs = st.session_state.sampling_rate
        preview_len = min(len(sig), fs * 10)
        fig = plot_signal(sig[:preview_len], fs,
                          title=t("preview_plot", st=st.session_state.signal_type))
        st.plotly_chart(fig, use_container_width=True)

        info_cols = st.columns(4)
        info_cols[0].metric(t("metric_samples"), f"{len(sig):,}")
        info_cols[1].metric(t("metric_freq"), f"{fs} Hz")
        info_cols[2].metric(t("metric_duration"), f"{len(sig)/fs:.1f} s")
        info_cols[3].metric(t("metric_type"), st.session_state.signal_type)

        st.success(t("step1_done"))
    else:
        st.info(t("no_signal"))

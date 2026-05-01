import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.i18n import t, render_language_selector
from components.signal_plot import plot_signal
from components.hrv_plots import plot_poincare, plot_hrv_spectrum, plot_hrv_psd
from components.report_pdf import generate_pdf_report

st.set_page_config(page_title="Step 5 — Report", page_icon="📋", layout="wide")

render_language_selector()

st.title(t("step5_title"))

if st.session_state.get("biomarker_report") is None:
    st.warning(t("go_step4"))
    st.stop()

report = st.session_state.biomarker_report
signal_type = st.session_state.signal_type
_clean = st.session_state.get("clean_signal")
sig = _clean if _clean is not None else st.session_state.signal_data
fs = st.session_state.sampling_rate
now = datetime.now().strftime("%Y-%m-%d %H:%M")

st.markdown(f"""
---
{t("report_meta", now=now, st=signal_type, dur=report.get("duration_seconds", 0), peaks=report.get("n_peaks_detected", 0))}

> {t("report_disclaimer")}
---
""")

# Status label map
status_map = {
    "normal":     t("status_normal"),
    "borderline": t("status_borderline"),
    "abnormal":   t("status_abnormal"),
    "unknown":    t("status_unknown"),
}

all_bms = []
for section_key in ["hrv_time", "hrv_freq", "ecg_morphology", "ppg_vascular", "hrv_nonlinear"]:
    section = report.get(section_key)
    if section:
        for bm_key, bm in section.items():
            if isinstance(bm, dict) and "name" in bm:
                all_bms.append({
                    t("col_biomarker"): bm["name"],
                    t("col_value"):     f"{bm['value']:.3f}" if bm["value"] is not None else t("val_na"),
                    t("col_unit"):      bm["unit"],
                    t("col_normal_range"): f"{bm['normal_range'][0]}–{bm['normal_range'][1]}",
                    t("col_status"):    status_map.get(bm["risk_level"], "?"),
                })

if all_bms:
    df_report = pd.DataFrame(all_bms)
    status_col = t("col_status")

    def color_rows(row):
        color_map = {
            t("status_normal"):     "background-color: #d4edda",
            t("status_borderline"): "background-color: #fff3cd",
            t("status_abnormal"):   "background-color: #f8d7da",
            t("status_unknown"):    "",
        }
        return [color_map.get(row[status_col], "")] * len(row)

    st.subheader(t("sec_biomarker_table"))
    st.dataframe(df_report.style.apply(color_rows, axis=1), use_container_width=True)

ml = report.get("ml_anomaly")
if ml:
    st.subheader(t("sec_ml"))
    col_a, col_b, col_c = st.columns(3)
    col_a.metric(t("ml_detected"), t("ml_yes") if ml["is_anomalous"] else t("ml_no"))
    col_b.metric(t("ml_score"), f"{ml['anomaly_score']:.4f}")
    col_c.metric(t("ml_confidence"), f"{ml['confidence']*100:.0f}%")
    if ml.get("flags"):
        st.markdown(t("ml_flags"))
        for f in ml["flags"]:
            st.markdown(f"  - ⚠️ {f}")

st.subheader(t("sec_signal_plot"))
preview_len = min(len(sig), fs * 15)
peaks = st.session_state.get("peaks")
peaks_preview = peaks[peaks < preview_len] if peaks is not None else None
fig = plot_signal(
    sig[:preview_len], fs,
    title=t("plot_preprocessed", st=signal_type),
    peaks=peaks_preview,
    color="#e74c3c" if signal_type == "ECG" else "#3498db",
)
st.plotly_chart(fig, use_container_width=True)

if report.get("hrv_freq"):
    hrv_f = report["hrv_freq"]
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if peaks is not None and len(peaks) >= 10:
            rr_ms = np.diff(peaks) / fs * 1000
            st.plotly_chart(plot_poincare(rr_ms.tolist()), use_container_width=True)
    with col_b:
        vlf = hrv_f["vlf_power"]["value"] or 0
        lf = hrv_f["lf_power"]["value"] or 0
        hf = hrv_f["hf_power"]["value"] or 0
        st.plotly_chart(plot_hrv_spectrum(vlf, lf, hf), use_container_width=True)
    with col_c:
        if peaks is not None and len(peaks) >= 10:
            rr_ms = np.diff(peaks) / fs * 1000
            st.plotly_chart(plot_hrv_psd(rr_ms.tolist()), use_container_width=True)

if report.get("signal_quality"):
    from components.biomarker_panel import render_sqi_widget
    st.subheader(t("sec_sqi"))
    render_sqi_widget(report["signal_quality"])

if report.get("arrhythmia"):
    from components.biomarker_panel import render_arrhythmia_panel
    render_arrhythmia_panel(report["arrhythmia"])

if report.get("time_freq") and report["time_freq"].get("stft"):
    from components.hrv_plots import plot_stft_heatmap, plot_lf_hf_over_time
    st.subheader(t("sec_stft"))
    st.plotly_chart(plot_stft_heatmap(report["time_freq"]["stft"]), use_container_width=True)
    st.plotly_chart(plot_lf_hf_over_time(report["time_freq"]["stft"]), use_container_width=True)

st.subheader(t("sec_export"))
col_a, col_b, col_c = st.columns(3)

with col_a:
    if all_bms:
        csv = df_report.to_csv(index=False)
        st.download_button(
            label=t("btn_csv"),
            data=csv,
            file_name=f"report_{signal_type}_{now.replace(' ', '_').replace(':', '')}.csv",
            mime="text/csv",
        )

with col_b:
    json_report = json.dumps(report, indent=2, default=str)
    st.download_button(
        label=t("btn_json"),
        data=json_report,
        file_name=f"report_{signal_type}_{now.replace(' ', '_').replace(':', '')}.json",
        mime="application/json",
    )

with col_c:
    try:
        pdf_bytes = generate_pdf_report(report, sig, fs, peaks, signal_type)
        st.download_button(
            label=t("btn_pdf"),
            data=pdf_bytes,
            file_name=f"report_{signal_type}_{now.replace(' ', '_').replace(':', '')}.pdf",
            mime="application/pdf",
        )
    except Exception as _pdf_err:
        st.error(t("pdf_err", err=_pdf_err))

st.markdown("---")
st.info(t("new_analysis"))

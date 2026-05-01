import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.signal_plot import plot_signal
from components.hrv_plots import plot_poincare, plot_hrv_spectrum

st.set_page_config(page_title="Step 5 — Report", page_icon="📋", layout="wide")
st.title("📋 Step 5: Report Clinico")

if st.session_state.get("biomarker_report") is None:
    st.warning("Vai prima allo Step 4 per estrarre i biomarker.")
    st.stop()

report = st.session_state.biomarker_report
signal_type = st.session_state.signal_type
sig = st.session_state.get("clean_signal") or st.session_state.signal_data
fs = st.session_state.sampling_rate
now = datetime.now().strftime("%Y-%m-%d %H:%M")

st.markdown(f"""
---
**Data analisi:** {now} | **Tipo segnale:** {signal_type} | **Durata:** {report.get('duration_seconds', 0):.1f}s | **Picchi rilevati:** {report.get('n_peaks_detected', 0)}

> ⚠️ *Questo report è generato a scopo di ricerca. Non costituisce diagnosi medica.*
---
""")

# Summary table
all_bms = []
for section_key in ["hrv_time", "hrv_freq", "ecg_morphology", "ppg_vascular"]:
    section = report.get(section_key)
    if section:
        for bm_key, bm in section.items():
            if isinstance(bm, dict) and "name" in bm:
                all_bms.append({
                    "Biomarker": bm["name"],
                    "Valore": f"{bm['value']:.3f}" if bm["value"] is not None else "N/D",
                    "Unità": bm["unit"],
                    "Range Normale": f"{bm['normal_range'][0]}–{bm['normal_range'][1]}",
                    "Stato": {"normal": "✓ Normale", "borderline": "⚠ Borderline", "abnormal": "✗ Anomalo", "unknown": "? N/D"}.get(bm["risk_level"], "?"),
                })

if all_bms:
    df_report = pd.DataFrame(all_bms)

    def color_rows(row):
        color_map = {"✓ Normale": "background-color: #d4edda", "⚠ Borderline": "background-color: #fff3cd", "✗ Anomalo": "background-color: #f8d7da", "? N/D": ""}
        return [color_map.get(row["Stato"], "")] * len(row)

    st.subheader("📊 Tabella Biomarker Completa")
    st.dataframe(df_report.style.apply(color_rows, axis=1), use_container_width=True)

# ML Summary
ml = report.get("ml_anomaly")
if ml:
    st.subheader("🤖 ML Anomaly Detection")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Anomalia rilevata", "SÌ" if ml["is_anomalous"] else "NO")
    col_b.metric("Anomaly Score", f"{ml['anomaly_score']:.4f}")
    col_c.metric("Confidenza", f"{ml['confidence']*100:.0f}%")
    if ml.get("flags"):
        st.markdown("**Flag clinici:**")
        for f in ml["flags"]:
            st.markdown(f"  - ⚠️ {f}")

# Signal plot
st.subheader("📈 Segnale Analizzato")
preview_len = min(len(sig), fs * 15)
peaks = st.session_state.get("peaks")
peaks_preview = peaks[peaks < preview_len] if peaks is not None else None
fig = plot_signal(sig[:preview_len], fs,
                 title=f"{signal_type} preprocessato",
                 peaks=peaks_preview,
                 color="#e74c3c" if signal_type == "ECG" else "#3498db")
st.plotly_chart(fig, use_container_width=True)

# HRV plots
if report.get("hrv_freq"):
    hrv_f = report["hrv_freq"]
    col_a, col_b = st.columns(2)
    with col_a:
        if peaks is not None and len(peaks) >= 10:
            rr_ms = np.diff(peaks) / fs * 1000
            fig_p = plot_poincare(rr_ms.tolist())
            st.plotly_chart(fig_p, use_container_width=True)
    with col_b:
        vlf = hrv_f["vlf_power"]["value"] or 0
        lf = hrv_f["lf_power"]["value"] or 0
        hf = hrv_f["hf_power"]["value"] or 0
        fig_s = plot_hrv_spectrum(vlf, lf, hf)
        st.plotly_chart(fig_s, use_container_width=True)

# Export
st.subheader("💾 Export Report")
col_a, col_b = st.columns(2)

with col_a:
    if all_bms:
        csv = df_report.to_csv(index=False)
        st.download_button(
            label="⬇️ Scarica CSV",
            data=csv,
            file_name=f"report_{signal_type}_{now.replace(' ', '_').replace(':', '')}.csv",
            mime="text/csv",
        )

with col_b:
    json_report = json.dumps(report, indent=2, default=str)
    st.download_button(
        label="⬇️ Scarica JSON",
        data=json_report,
        file_name=f"report_{signal_type}_{now.replace(' ', '_').replace(':', '')}.json",
        mime="application/json",
    )

st.markdown("---")
st.info("**Nuova analisi?** Torna allo **Step 1** per caricare un nuovo segnale.")

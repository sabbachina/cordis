import streamlit as st
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.api_client import analyze_signal, check_backend_health
from components.biomarker_panel import render_biomarker_section
from components.hrv_plots import plot_poincare, plot_hrv_spectrum

st.set_page_config(page_title="Step 4 — Biomarker", page_icon="🧬", layout="wide")
st.title("🧬 Step 4: Estrazione Digital Biomarker")

if st.session_state.get("signal_data") is None:
    st.warning("Vai prima allo Step 1.")
    st.stop()
if st.session_state.get("peaks") is None:
    st.warning("Vai prima allo Step 3 per rilevare i picchi.")
    st.stop()

sig = st.session_state.get("clean_signal") or st.session_state.signal_data
fs = st.session_state.sampling_rate
signal_type = st.session_state.signal_type
prep = st.session_state.get("preprocessing_config", {"lowcut": 0.5, "highcut": 40.0, "method": "butterworth", "remove_baseline": True})

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Selezione Biomarker")
    compute_hrv = st.checkbox("HRV (Heart Rate Variability)", value=True)
    compute_morphology = st.checkbox("Morfologia segnale", value=True,
                                     help="Per ECG: QTc, PR, QRS, ST. Per PPG: AIx, Resp Rate.")
    compute_ml = st.checkbox("ML Anomaly Detection", value=True,
                             help="IsolationForest + regole cliniche per flagging automatico")

    min_duration = len(sig) / fs
    if min_duration < 30 and compute_hrv:
        st.warning(f"⚠️ Segnale di {min_duration:.0f}s: HRV frequenziale richiede ≥ 30s per risultati affidabili.")

    backend_ok = check_backend_health()

    if st.button("🧬 Estrai Biomarker", type="primary"):
        if not backend_ok:
            st.error("❌ Backend non raggiungibile. Avvia il backend con: `uvicorn main:app --reload`")
        else:
            payload = {
                "signal": {
                    "signal_type": signal_type,
                    "sampling_rate": int(fs),
                    "values": sig.tolist(),
                },
                "preprocessing": prep,
                "compute_hrv": compute_hrv,
                "compute_morphology": compute_morphology,
                "compute_ml": compute_ml,
            }
            with st.spinner("Analisi in corso (può richiedere 10-30s)..."):
                try:
                    report = analyze_signal(payload)
                    st.session_state.biomarker_report = report
                    st.success("✅ Biomarker estratti con successo!")
                except Exception as e:
                    st.error(f"Errore analisi: {e}")

with col2:
    report = st.session_state.get("biomarker_report")
    if report:
        # Warnings
        if report.get("warnings"):
            for w in report["warnings"]:
                st.warning(f"⚠️ {w}")

        # ML Anomaly
        ml = report.get("ml_anomaly")
        if ml:
            if ml["is_anomalous"]:
                st.error(f"🚨 **ANOMALIA RILEVATA** (score: {ml['anomaly_score']:.3f}, confidenza: {ml['confidence']*100:.0f}%)")
                for flag in ml.get("flags", []):
                    st.error(f"  • {flag}")
            else:
                st.success(f"✅ Nessuna anomalia ML (score: {ml['anomaly_score']:.3f})")

        # HRV Temporale
        if report.get("hrv_time"):
            hrv_t = report["hrv_time"]
            bms = [hrv_t["mean_hr"], hrv_t["sdnn"], hrv_t["rmssd"], hrv_t["pnn50"], hrv_t["pnn20"]]
            render_biomarker_section("📊 HRV — Dominio Temporale", bms)

        # HRV Frequenziale
        if report.get("hrv_freq"):
            hrv_f = report["hrv_freq"]
            bms = [hrv_f["vlf_power"], hrv_f["lf_power"], hrv_f["hf_power"], hrv_f["lf_hf_ratio"]]
            render_biomarker_section("📈 HRV — Dominio Frequenziale", bms)

            vlf = hrv_f["vlf_power"]["value"] or 0
            lf = hrv_f["lf_power"]["value"] or 0
            hf = hrv_f["hf_power"]["value"] or 0
            fig_spec = plot_hrv_spectrum(vlf, lf, hf)
            st.plotly_chart(fig_spec, use_container_width=True)

        # Morfologia ECG
        if report.get("ecg_morphology"):
            morph = report["ecg_morphology"]
            bms = [morph["pr_interval_ms"], morph["qrs_duration_ms"], morph["qtc_ms"], morph["st_deviation_mv"]]
            render_biomarker_section("❤️ Morfologia ECG", bms)

        # Parametri PPG
        if report.get("ppg_vascular"):
            ppg_v = report["ppg_vascular"]
            bms = [ppg_v["pulse_amplitude"], ppg_v["augmentation_index"], ppg_v["respiratory_rate"]]
            render_biomarker_section("🩺 Parametri Vascolari PPG", bms)

        # Poincaré
        peaks = st.session_state.get("peaks")
        if peaks is not None and len(peaks) >= 10:
            rr_ms = np.diff(peaks) / fs * 1000
            st.subheader("🔵 Poincaré Plot HRV")
            fig_p = plot_poincare(rr_ms.tolist())
            st.plotly_chart(fig_p, use_container_width=True)

        st.success("✅ Vai al **Step 5: Report** →")
    else:
        st.info("Clicca 'Estrai Biomarker' per iniziare l'analisi.")

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

_clean = st.session_state.get("clean_signal")
sig = _clean if _clean is not None else st.session_state.signal_data
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
    compute_nonlinear = st.checkbox("HRV Non-lineare (Poincaré, DFA, Entropy)", value=True)
    compute_arrhythmia = st.checkbox("Analisi Aritmia (AFib, ectopici)", value=True)
    compute_signal_quality = st.checkbox("Signal Quality Index", value=True)

    st.markdown("---")
    st.subheader("🔬 Kubios Scientific")
    compute_advanced = st.checkbox("HRV Avanzato (DFA α2, ApEn, FuzzyEn, MSE, RQA, LLE)", value=True,
                                   help="Richiede ≥ 128 battiti per DFA α2; ≥ 10 battiti per gli altri.")
    compute_autonomic = st.checkbox("Indici Autonomici (PNS, SNS, Baevsky)", value=True,
                                    help="Parasympathetic/Sympathetic index e Baevsky Stress Index.")
    compute_timefreq = st.checkbox("Analisi Tempo-Frequenza (STFT/CWT)", value=False,
                                   help="Computazionalmente costoso. Richiede segnale ≥ 30s.")
    artifact_correction = st.checkbox("Correzione Artefatti RR (Kubios-like)", value=True,
                                      help="Rilevamento e interpolazione spline degli artefatti negli intervalli RR.")

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
                "compute_nonlinear": compute_nonlinear,
                "compute_arrhythmia": compute_arrhythmia,
                "compute_signal_quality": compute_signal_quality,
                "compute_advanced": compute_advanced,
                "compute_autonomic": compute_autonomic,
                "compute_timefreq": compute_timefreq,
                "artifact_correction": artifact_correction,
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

        # Artifact correction summary
        ac = report.get("artifact_correction")
        if ac:
            if ac["n_artifacts"] > 0:
                st.info(f"🔧 **Correzione artefatti**: {ac['n_artifacts']} su {ac['n_total_beats']} battiti "
                        f"({ac['artifact_pct']:.1f}%) — qualità: **{ac['quality_label']}**")
            else:
                st.success(f"✅ Nessun artefatto RR rilevato ({ac['n_total_beats']} battiti)")

        # HRV Temporale (esteso Kubios)
        if report.get("hrv_time"):
            hrv_t = report["hrv_time"]
            base_bms = [hrv_t["mean_hr"], hrv_t["sdnn"], hrv_t["rmssd"], hrv_t["pnn50"], hrv_t["pnn20"]]
            kubios_bms = [hrv_t[k] for k in ("nn50", "sdann", "sdnni", "hrvi", "tinn") if hrv_t.get(k)]
            render_biomarker_section("📊 HRV — Dominio Temporale", base_bms + kubios_bms)

        # HRV Frequenziale (esteso Kubios)
        if report.get("hrv_freq"):
            hrv_f = report["hrv_freq"]
            base_bms = [hrv_f["vlf_power"], hrv_f["lf_power"], hrv_f["hf_power"], hrv_f["lf_hf_ratio"]]
            kubios_bms = [hrv_f[k] for k in ("total_power", "lfnu", "hfnu") if hrv_f.get(k)]
            render_biomarker_section("📈 HRV — Dominio Frequenziale", base_bms + kubios_bms)

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

        # HRV Non-lineare
        if report.get("hrv_nonlinear"):
            from components.biomarker_panel import render_nonlinear_section
            render_nonlinear_section(report["hrv_nonlinear"])

        # HRV Avanzato Kubios
        if report.get("hrv_advanced"):
            adv = report["hrv_advanced"]
            adv_bms = [adv[k] for k in (
                "dfa_alpha2", "approximate_entropy", "fuzzy_entropy",
                "mse_slope", "rqa_rr_pct", "rqa_det", "rqa_entr", "lyapunov_exponent"
            ) if adv.get(k)]
            if adv_bms:
                render_biomarker_section("🔬 HRV Avanzato (Kubios Scientific)", adv_bms)

        # Autonomic Indices
        if report.get("autonomic"):
            auto = report["autonomic"]
            auto_bms = [auto[k] for k in ("pns_index", "sns_index", "baevsky_stress_index", "autonomic_balance") if auto.get(k)]
            if auto_bms:
                render_biomarker_section("🫀 Indici Autonomici (SNS/PNS)", auto_bms)

        # Time-Frequency summary
        if report.get("time_freq"):
            tf = report["time_freq"]
            st.subheader("📡 Analisi Tempo-Frequenza")
            c1, c2, c3 = st.columns(3)
            c1.metric("STFT", "✅" if tf["has_stft"] else "—")
            c2.metric("CWT Morlet", "✅" if tf["has_cwt"] else "—")
            if tf.get("lf_hf_variability") is not None:
                c3.metric("Variabilità LF/HF", f"{tf['lf_hf_variability']:.3f}")
            if tf.get("dominant_lf_time_pct") is not None:
                st.info(f"Dominanza LF (simpatica) nel {tf['dominant_lf_time_pct']:.1f}% del tempo.")

        # Aritmia
        if report.get("arrhythmia"):
            from components.biomarker_panel import render_arrhythmia_panel
            render_arrhythmia_panel(report["arrhythmia"])

        # Signal Quality
        if report.get("signal_quality"):
            from components.biomarker_panel import render_sqi_widget
            render_sqi_widget(report["signal_quality"])

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

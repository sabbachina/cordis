import streamlit as st
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.i18n import t, render_language_selector
from utils.api_client import analyze_signal, check_backend_health
from components.biomarker_panel import render_biomarker_section
from components.hrv_plots import plot_poincare, plot_hrv_spectrum

st.set_page_config(page_title="Step 4 — Biomarker", page_icon="", layout="wide")

render_language_selector()

st.title(t("step4_title"))

if st.session_state.get("signal_data") is None:
    st.warning(t("go_step1_warn"))
    st.stop()
if st.session_state.get("peaks") is None:
    st.warning(t("go_step3"))
    st.stop()

_clean = st.session_state.get("clean_signal")
sig = _clean if _clean is not None else st.session_state.signal_data
fs = st.session_state.sampling_rate
signal_type = st.session_state.signal_type
prep = st.session_state.get("preprocessing_config", {"lowcut": 0.5, "highcut": 40.0, "method": "butterworth", "remove_baseline": True})

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader(t("biomarker_select"))
    compute_hrv = st.checkbox(t("chk_hrv"), value=True)
    compute_morphology = st.checkbox(t("chk_morphology"), value=True, help=t("chk_morphology_help"))
    compute_ml = st.checkbox(t("chk_ml"), value=True, help=t("chk_ml_help"))
    compute_nonlinear = st.checkbox(t("chk_nonlinear"), value=True)
    compute_arrhythmia = st.checkbox(t("chk_arrhythmia"), value=True)
    compute_signal_quality = st.checkbox(t("chk_sqi"), value=True)

    st.markdown("---")
    st.subheader(t("hrv_advanced_section"))
    compute_advanced = st.checkbox(t("chk_advanced"), value=True, help=t("chk_advanced_help"))
    compute_autonomic = st.checkbox(t("chk_autonomic"), value=True, help=t("chk_autonomic_help"))
    compute_timefreq = st.checkbox(t("chk_timefreq"), value=False, help=t("chk_timefreq_help"))
    artifact_correction = st.checkbox(t("chk_artifact"), value=True, help=t("chk_artifact_help"))

    min_duration = len(sig) / fs
    if min_duration < 30 and compute_hrv:
        st.warning(t("short_signal_warn", dur=min_duration))

    backend_ok = check_backend_health()

    if st.button(t("btn_extract"), type="primary"):
        if not backend_ok:
            st.error(t("backend_unreachable"))
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
            with st.spinner(t("spinner_extract")):
                try:
                    report = analyze_signal(payload)
                    st.session_state.biomarker_report = report
                    st.success(t("extract_ok"))
                except Exception as e:
                    st.error(t("extract_err", err=e))

with col2:
    report = st.session_state.get("biomarker_report")
    if report:
        if report.get("warnings"):
            for w in report["warnings"]:
                st.warning(f"⚠ {w}")

        ml = report.get("ml_anomaly")
        if ml:
            if ml["is_anomalous"]:
                st.error(f" **{t('ml_detected')}** (score: {ml['anomaly_score']:.3f}, {t('ml_confidence')}: {ml['confidence']*100:.0f}%)")
                for flag in ml.get("flags", []):
                    st.error(f"  • {flag}")
            else:
                st.success(f" {t('ml_no')} (score: {ml['anomaly_score']:.3f})")

        ac = report.get("artifact_correction")
        if ac:
            if ac["n_artifacts"] > 0:
                st.info(t("artifact_fixed", n=ac["n_artifacts"], tot=ac["n_total_beats"],
                          pct=ac["artifact_pct"], q=ac["quality_label"]))
            else:
                st.success(t("artifact_ok", n=ac["n_total_beats"]))

        if report.get("hrv_time"):
            hrv_t = report["hrv_time"]
            base_bms = [hrv_t["mean_hr"], hrv_t["sdnn"], hrv_t["rmssd"], hrv_t["pnn50"], hrv_t["pnn20"]]
            ext_bms = [hrv_t[k] for k in ("nn50", "sdann", "sdnni", "hrvi", "tinn") if hrv_t.get(k)]
            render_biomarker_section(t("sec_hrv_time"), base_bms + ext_bms)

        if report.get("hrv_freq"):
            hrv_f = report["hrv_freq"]
            base_bms = [hrv_f["vlf_power"], hrv_f["lf_power"], hrv_f["hf_power"], hrv_f["lf_hf_ratio"]]
            ext_bms = [hrv_f[k] for k in ("total_power", "lfnu", "hfnu") if hrv_f.get(k)]
            render_biomarker_section(t("sec_hrv_freq"), base_bms + ext_bms)

            vlf = hrv_f["vlf_power"]["value"] or 0
            lf = hrv_f["lf_power"]["value"] or 0
            hf = hrv_f["hf_power"]["value"] or 0
            st.plotly_chart(plot_hrv_spectrum(vlf, lf, hf), use_container_width=True)

        if report.get("ecg_morphology"):
            morph = report["ecg_morphology"]
            bms = [morph["pr_interval_ms"], morph["qrs_duration_ms"], morph["qtc_ms"], morph["st_deviation_mv"]]
            render_biomarker_section(t("sec_ecg_morph"), bms)

        if report.get("ppg_vascular"):
            ppg_v = report["ppg_vascular"]
            bms = [ppg_v["pulse_amplitude"], ppg_v["augmentation_index"], ppg_v["respiratory_rate"]]
            render_biomarker_section(t("sec_ppg_vasc"), bms)

        if report.get("hrv_nonlinear"):
            from components.biomarker_panel import render_nonlinear_section
            render_nonlinear_section(report["hrv_nonlinear"])

        if report.get("hrv_advanced"):
            adv = report["hrv_advanced"]
            adv_bms = [adv[k] for k in (
                "dfa_alpha2", "approximate_entropy", "fuzzy_entropy",
                "mse_slope", "rqa_rr_pct", "rqa_det", "rqa_entr", "lyapunov_exponent"
            ) if adv.get(k)]
            if adv_bms:
                render_biomarker_section(t("sec_hrv_adv"), adv_bms)

        if report.get("autonomic"):
            auto = report["autonomic"]
            auto_bms = [auto[k] for k in ("pns_index", "sns_index", "baevsky_stress_index", "autonomic_balance") if auto.get(k)]
            if auto_bms:
                render_biomarker_section(t("sec_autonomic"), auto_bms)

        if report.get("time_freq"):
            tf = report["time_freq"]
            st.subheader(t("sec_timefreq"))
            c1, c2, c3 = st.columns(3)
            c1.metric(t("tf_stft"), "" if tf["has_stft"] else "—")
            c2.metric(t("tf_cwt"), "" if tf["has_cwt"] else "—")
            if tf.get("lf_hf_variability") is not None:
                c3.metric(t("tf_lf_hf_var"), f"{tf['lf_hf_variability']:.3f}")
            if tf.get("dominant_lf_time_pct") is not None:
                st.info(t("tf_lf_dominance", pct=tf["dominant_lf_time_pct"]))

        if report.get("arrhythmia"):
            from components.biomarker_panel import render_arrhythmia_panel
            render_arrhythmia_panel(report["arrhythmia"])

        if report.get("signal_quality"):
            from components.biomarker_panel import render_sqi_widget
            render_sqi_widget(report["signal_quality"])

        peaks = st.session_state.get("peaks")
        if peaks is not None and len(peaks) >= 10:
            rr_ms = np.diff(peaks) / fs * 1000
            st.subheader(t("poincare_title"))
            st.plotly_chart(plot_poincare(rr_ms.tolist()), use_container_width=True)

        st.success(t("step4_done"))
    else:
        st.info(t("extract_hint"))

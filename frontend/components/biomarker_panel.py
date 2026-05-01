import streamlit as st

RISK_COLORS = {"normal": "#2ecc71", "borderline": "#f39c12", "abnormal": "#e74c3c", "unknown": "#95a5a6"}
RISK_LABELS = {"normal": "✓ Normale", "borderline": "⚠ Borderline", "abnormal": "✗ Anomalo", "unknown": "? N/D"}

def render_biomarker_card(name: str, value, unit: str, risk_level: str, description: str, normal_range: tuple):
    color = RISK_COLORS.get(risk_level, "#95a5a6")
    label = RISK_LABELS.get(risk_level, "?")
    val_str = f"{value:.2f}" if value is not None else "N/D"
    st.markdown(f"""
    <div style="border-left: 4px solid {color}; padding: 8px 12px; margin: 4px 0; background: #f8f9fa; border-radius: 4px;">
        <b>{name}</b><br/>
        <span style="font-size: 1.4em; font-weight: bold; color: {color};">{val_str} {unit}</span>
        &nbsp;&nbsp;<span style="font-size: 0.85em; color: {color};">{label}</span><br/>
        <span style="font-size: 0.8em; color: #666;">Range normale: {normal_range[0]}–{normal_range[1]} {unit} | {description}</span>
    </div>
    """, unsafe_allow_html=True)

def render_biomarker_section(title: str, biomarkers: list):
    st.subheader(title)
    cols = st.columns(2)
    for i, bm in enumerate(biomarkers):
        with cols[i % 2]:
            render_biomarker_card(
                name=bm.get("name", ""),
                value=bm.get("value"),
                unit=bm.get("unit", ""),
                risk_level=bm.get("risk_level", "unknown"),
                description=bm.get("description", ""),
                normal_range=bm.get("normal_range", (0, 0)),
            )


def render_arrhythmia_panel(arrhythmia: dict):
    """Pannello dedicato aritmia con badge colorati."""
    hr_class = arrhythmia.get("heart_rate_class", "normal")
    afib = arrhythmia.get("afib_suspected", False)
    ectopic = arrhythmia.get("ectopic_beats", 0)
    ectopic_ratio = arrhythmia.get("ectopic_ratio", 0.0)
    rr_cv = arrhythmia.get("rr_cv", 0.0)

    st.subheader("Analisi Aritmia")

    col1, col2, col3, col4 = st.columns(4)

    hr_labels = {"normal": "✓ Normale", "bradycardia": "⚠ Bradicardia", "tachycardia": "⚠ Tachicardia"}
    col1.metric("Classificazione HR", hr_labels.get(hr_class, hr_class))

    afib_str = "⚠️ Sospetta AFib" if afib else "✓ Ritmo regolare"
    col2.metric("Pattern Ritmo", afib_str)

    col3.metric("Battiti Ectopici", f"{ectopic} ({ectopic_ratio*100:.1f}%)",
                help="Battiti con RR > ±20% rispetto al precedente")
    col4.metric("CV intervalli RR", f"{rr_cv:.3f}",
                help="Coefficiente di variazione RR (>0.20 = alta irregolarità)")

    if arrhythmia.get("afib_evidence"):
        st.warning("**Evidenze per sospetta FA:**\n" + "\n".join(f"- {e}" for e in arrhythmia["afib_evidence"]))

    if ectopic_ratio > 0.05:
        st.warning(f"⚠️ {ectopic_ratio*100:.1f}% di battiti ectopici rilevati — valutare ulteriormente")


def render_sqi_widget(sqi: dict):
    """Gauge Signal Quality Index con metriche dettagliate."""
    score = sqi.get("overall_score", 0)
    label = sqi.get("quality_label", "Unknown")

    colors = {"Good": "#2ecc71", "Acceptable": "#f39c12", "Poor": "#e67e22", "Unacceptable": "#e74c3c"}
    color = colors.get(label, "#95a5a6")

    st.markdown(f"""
    <div style="border: 2px solid {color}; border-radius: 8px; padding: 12px; margin-bottom: 12px; background: rgba(0,0,0,0.02);">
        <b>Signal Quality Index (SQI)</b><br/>
        <span style="font-size: 2em; font-weight: bold; color: {color};">{score:.0f}/100</span>
        &nbsp;&nbsp;<span style="font-size: 1.1em; color: {color};">&#9679; {label}</span>
        <div style="background: #eee; border-radius: 4px; height: 10px; margin-top: 8px;">
            <div style="background: {color}; width: {score}%; height: 100%; border-radius: 4px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    cols[0].metric("SNR", f"{sqi.get('snr_db', 0):.1f} dB")
    cols[1].metric("Flatline", f"{sqi.get('flatline_fraction', 0)*100:.1f}%")
    cols[2].metric("Clipping", f"{sqi.get('clipping_fraction', 0)*100:.1f}%")
    cols[3].metric("Baseline", f"{sqi.get('baseline_wander_fraction', 0)*100:.1f}%")

    for w in sqi.get("warnings", []):
        st.warning(f"⚠️ {w}")


def render_nonlinear_section(nlr: dict):
    """Cards per biomarker HRV non-lineari."""
    bms = []
    for key in ["sd1", "sd2", "sd1_sd2_ratio", "sample_entropy", "dfa_alpha1"]:
        v = nlr.get(key)
        if isinstance(v, dict):
            bms.append(v)
    if bms:
        render_biomarker_section("🔬 HRV Non-Lineare (Complessità)", bms)

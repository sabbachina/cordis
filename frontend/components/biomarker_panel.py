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

def render_biomarker_section(title: str, biomarkers: list[dict]):
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

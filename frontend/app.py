import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.i18n import t, render_language_selector

st.set_page_config(
    page_title="ECG/PPG Analysis Platform",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session state initialization
if "signal_data" not in st.session_state:
    st.session_state.signal_data = None
if "signal_type" not in st.session_state:
    st.session_state.signal_type = None
if "sampling_rate" not in st.session_state:
    st.session_state.sampling_rate = None
if "clean_signal" not in st.session_state:
    st.session_state.clean_signal = None
if "peaks" not in st.session_state:
    st.session_state.peaks = None
if "biomarker_report" not in st.session_state:
    st.session_state.biomarker_report = None
if "lang" not in st.session_state:
    st.session_state.lang = "it"

st.sidebar.title(t("sidebar_title"))
render_language_selector()
st.sidebar.markdown("---")

steps = {
    t("nav_step1"): t("nav_desc1"),
    t("nav_step2"): t("nav_desc2"),
    t("nav_step3"): t("nav_desc3"),
    t("nav_step4"): t("nav_desc4"),
    t("nav_step5"): t("nav_desc5"),
    t("nav_step6"): t("nav_desc6"),
}

st.sidebar.markdown(f"### {t('sidebar_steps_header')}")
for step, desc in steps.items():
    st.sidebar.markdown(f"**{step}**")
    st.sidebar.caption(desc)

st.sidebar.markdown("---")
st.sidebar.markdown(f"⚠️ *{t('research_only')}*")
st.sidebar.markdown(f"*{t('not_ce')}*")

st.title(t("app_title"))
st.markdown(t("home_welcome"))

st.markdown(t("home_how"))
st.markdown(t("home_nav_hint"))

st.markdown(f"""
| {t('home_col_step')} | {t('home_col_page')} | {t('home_col_desc')} |
|------|--------|-------------|
| 1 | {t('home_row1_page')} | {t('home_row1_desc')} |
| 2 | {t('home_row2_page')} | {t('home_row2_desc')} |
| 3 | {t('home_row3_page')} | {t('home_row3_desc')} |
| 4 | {t('home_row4_page')} | {t('home_row4_desc')} |
| 5 | {t('home_row5_page')} | {t('home_row5_desc')} |
""")

st.markdown(t("home_signals"))
st.markdown(t("home_ecg_desc"))
st.markdown(t("home_ppg_desc"))

st.markdown(t("home_formats"))
st.markdown(t("home_formats_list"))

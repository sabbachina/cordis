import streamlit as st

st.set_page_config(
    page_title="ECG/PPG Analysis Platform",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("❤️ ECG/PPG Platform")
st.sidebar.markdown("---")

steps = {
    "1️⃣ Upload Segnale": "Upload e caricamento del segnale ECG o PPG",
    "2️⃣ Preprocessing": "Filtraggio e pulizia del segnale",
    "3️⃣ Peak Detection": "Rilevamento automatico dei picchi",
    "4️⃣ Biomarker": "Estrazione digital biomarker clinici",
    "5️⃣ Report": "Report clinico e export",
}

st.sidebar.markdown("### Wizard Steps")
for step, desc in steps.items():
    st.sidebar.markdown(f"**{step}**")
    st.sidebar.caption(desc)

st.sidebar.markdown("---")
st.sidebar.markdown("⚠️ *FOR RESEARCH USE ONLY*")
st.sidebar.markdown("*NOT CE-MARKED*")

st.title("🫀 ECG/PPG Analysis Platform")
st.markdown("""
Benvenuto nella piattaforma guidata per l'analisi di segnali ECG e PPG con estrazione di **digital biomarker clinici**.

### Come procedere:
Usa il **menu a sinistra** (pagine) per navigare attraverso i 5 step del wizard:

| Step | Pagina | Descrizione |
|------|--------|-------------|
| 1 | **Upload** | Carica il tuo file ECG o PPG (CSV, EDF, WFDB, JSON) oppure usa i dati campione |
| 2 | **Preprocessing** | Applica filtri e rimuovi artefatti dal segnale |
| 3 | **Peak Detection** | Individua automaticamente i picchi R (ECG) o sistolici (PPG) |
| 4 | **Biomarker** | Estrai HRV, QTc, Augmentation Index e altri biomarker clinici |
| 5 | **Report** | Visualizza e scarica il report clinico completo |

### Segnali supportati:
- **ECG** — Elettrocardiogramma (singola derivazione o multiderivazione)
- **PPG** — Fotopletismografia (da wearable o sensore clinico)

### Formati input:
CSV · Excel · EDF/EDF+ · WFDB/MIT-BIH · JSON
""")

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

# ECG/PPG Analysis Platform

![Python](https://img.shields.io/badge/Python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green) ![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red) ![License](https://img.shields.io/badge/License-MIT-yellow)

> **FOR RESEARCH USE ONLY — NOT CE-MARKED MEDICAL DEVICE**

Piattaforma open-source per l'analisi guidata passo-passo di segnali ECG e PPG con estrazione automatica di digital biomarker d'interesse clinico.

## Features

- **Wizard 5-step**: Upload → Preprocessing → Peak Detection → Biomarker Extraction → Report
- **Formati input**: CSV, Excel, EDF/EDF+, WFDB/MIT-BIH, JSON stream
- **Segnali supportati**: ECG (singola derivazione o multiderivazione) e PPG (da wearable o clinico)
- **Biomarker ECG**: HRV temporale (SDNN, RMSSD, pNN50), HRV frequenziale (LF, HF, LF/HF), morfologici (QTc, PR, QRS, ST)
- **Biomarker PPG**: HRV, Augmentation Index, Respiratory Rate, Pulse Amplitude
- **ML Anomaly Detection**: IsolationForest + regole cliniche per flagging automatico
- **Export**: Report CSV e visualizzazioni interattive Plotly

## Quick Start (Docker)

```bash
git clone https://github.com/your-org/ecg-ppg-platform.git
cd ecg-ppg-platform
docker-compose up
```

Apri il browser su [http://localhost:8501](http://localhost:8501)

## Installazione Locale

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend (nuovo terminale)
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## Struttura

```
├── backend/          # FastAPI + signal processing engine
│   ├── core/         # Signal loader, preprocessor, ECG/PPG analyzers
│   ├── api/          # REST endpoints
│   └── models/       # Pydantic data models
├── frontend/         # Streamlit wizard 5-step
│   ├── pages/        # Una pagina per step
│   └── components/   # Grafici Plotly riutilizzabili
├── data/samples/     # Dataset ECG/PPG di esempio
└── tests/            # Test suite pytest
```

## Biomarker Estratti

| Biomarker | Segnale | Range Normale |
|-----------|---------|---------------|
| SDNN | ECG/PPG | 50 ± 16 ms |
| RMSSD | ECG/PPG | 42 ± 15 ms |
| pNN50 | ECG/PPG | 2–10% |
| LF/HF ratio | ECG/PPG | 1.5–2.0 |
| QTc | ECG | 360–440 ms |
| PR interval | ECG | 120–200 ms |
| QRS duration | ECG | 80–120 ms |
| Augmentation Index | PPG | 0–30% |
| Respiratory Rate | PPG | 12–20 rpm |

## Disclaimer

Questa piattaforma è sviluppata a scopo di **ricerca e formazione**. Non è un dispositivo medico certificato (CE/FDA). Non utilizzare per diagnosi cliniche senza supervisione medica qualificata.

## License

MIT License — see [LICENSE](LICENSE)

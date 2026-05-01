# Cordis

![Python](https://img.shields.io/badge/Python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green) ![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red) ![License](https://img.shields.io/badge/License-MIT-yellow)

> **FOR RESEARCH USE ONLY — NOT CE-MARKED MEDICAL DEVICE**

Open-source platform for guided step-by-step analysis of ECG and PPG signals with automatic extraction of 56 digital biomarkers of clinical interest, advanced HRV analysis, autonomic indices, time-frequency decomposition, and an AI-powered conversational assistant.

Available in Italian and English.

---

## Features

- **6-step wizard**: Upload → Preprocessing → Peak Detection → Biomarker Extraction → Report → AI Assistant
- **Input formats**: CSV, Excel, EDF/EDF+, WFDB/MIT-BIH, JSON stream
- **Signals supported**: ECG (single or multi-lead) and PPG (wearable or clinical)
- **56 biomarkers** extracted automatically (see full table below)
- **Advanced HRV analysis**: time-domain, frequency-domain, nonlinear, autonomic indices, time-frequency
- **Artifact correction**: threshold + quotient + moving-median detection; cubic spline interpolation
- **Signal Quality Index (SQI)**: SNR, flatline, clipping, baseline wander fractions
- **ML Anomaly Detection**: IsolationForest + clinical rule engine
- **AI Assistant**: conversational analysis assistant via Google Gemini API
- **Export**: CSV, JSON, PDF report (reportlab)
- **Multilingual**: Italian / English UI

---

## Quick Start (Docker)

```bash
git clone https://github.com/your-org/cordis.git
cd cordis
docker-compose up
```

Open the browser at [http://localhost:8501](http://localhost:8501)

---

## Local Installation

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend (new terminal)
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

Backend API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Project Structure

```
├── backend/
│   ├── core/
│   │   ├── signal_loader.py          # Multi-format input parser
│   │   ├── preprocessor.py           # Filtering, normalization
│   │   ├── ecg_analyzer.py           # ECG peak detection + HRV
│   │   ├── ppg_analyzer.py           # PPG peak detection + vascular indices
│   │   ├── biomarker_extractor.py    # Orchestrates all analyzers
│   │   ├── hrv_advanced.py           # DFA alpha2, ApEn, FuzzyEn, MSE, RQA, LLE
│   │   ├── hrv_autonomic.py          # PNS index, SNS index, Baevsky SI
│   │   ├── hrv_timefreq.py           # STFT and CWT Morlet time-frequency analysis
│   │   ├── artifact_correction.py    # RR artifact detection and interpolation
│   │   └── signal_quality.py         # SQI computation
│   ├── api/
│   │   └── endpoints/
│   │       ├── analysis.py           # POST /analysis/analyze
│   │       └── signals.py            # GET /signals/sample/{type}
│   └── models/
│       ├── biomarker.py              # All report Pydantic models
│       └── signal.py                 # Signal request/response models
├── frontend/
│   ├── app.py                        # Entry point, sidebar navigation
│   ├── pages/
│   │   ├── 1_Upload.py
│   │   ├── 2_Preprocessing.py
│   │   ├── 3_Peak_Detection.py
│   │   ├── 4_Feature_Extraction.py
│   │   ├── 5_Report.py
│   │   └── 6_Assistant.py            # AI conversational assistant
│   ├── components/
│   │   ├── signal_plot.py            # Raw/clean signal Plotly charts
│   │   ├── hrv_plots.py              # Poincare, PSD, STFT, CWT plots
│   │   ├── biomarker_panel.py        # Cards, SQI widget, arrhythmia panel
│   │   ├── report_pdf.py             # PDF report generation (reportlab)
│   │   └── ai_assistant.py           # GeminiAssistant, context builder
│   └── utils/
│       ├── i18n.py                   # 278-key translation dict IT/EN
│       └── api_client.py             # httpx calls to backend
├── data/samples/                     # Sample ECG/PPG datasets
└── tests/                            # pytest test suite
```

---

## Biomarkers (56 total)

### HRV Time-Domain (10)

| Biomarker | Description | Normal Range |
|-----------|-------------|--------------|
| SDNN | Std dev of all NN intervals | 50 ± 16 ms |
| RMSSD | Root mean square of successive differences | 42 ± 15 ms |
| pNN50 | Fraction of successive intervals > 50 ms | 2–10% |
| pNN20 | Fraction of successive intervals > 20 ms | — |
| NN50 | Absolute count of successive intervals > 50 ms | — |
| Mean HR | Mean heart rate | 60–100 bpm |
| SDANN | Std dev of 5-min segment means | 50 ± 13 ms |
| SDNNi | Mean of 5-min segment std deviations | 40 ± 15 ms |
| HRVi | Triangular index of RR histogram | 37 ± 15 |
| TINN | Triangular interpolation baseline width | 331 ± 127 ms |

### HRV Frequency-Domain (7)

| Biomarker | Description | Normal Range |
|-----------|-------------|--------------|
| VLF Power | Very low frequency power (0.003–0.04 Hz) | — ms² |
| LF Power | Low frequency power (0.04–0.15 Hz) | 519 ± 291 ms² |
| HF Power | High frequency power (0.15–0.4 Hz) | 657 ± 777 ms² |
| LF/HF ratio | Sympathovagal balance | 1.5–2.0 |
| Total Power | VLF + LF + HF | 1000–4000 ms² |
| LFnu | LF normalized units | 40–60 n.u. |
| HFnu | HF normalized units | 30–40 n.u. |

### HRV Nonlinear (11)

| Biomarker | Description | Normal Range |
|-----------|-------------|--------------|
| SD1 | Poincare short-axis (vagal activity) | 17 ± 9 ms |
| SD2 | Poincare long-axis (overall HRV) | 32 ± 13 ms |
| SD1/SD2 | Poincare ratio | 0.25–0.5 |
| Sample Entropy | Signal irregularity | 0.7–1.5 |
| DFA alpha1 | Short-range fractal scaling (4–16 beats) | 1.0–1.2 |
| DFA alpha2 | Long-range fractal scaling (16–64 beats) | 0.9–1.1 |
| Approx. Entropy | Signal complexity (less sensitive to length) | 0.7–1.5 |
| Fuzzy Entropy | Robust entropy estimate for short series | — |
| MSE (curve) | Multiscale entropy (scales 1–20) | — |
| RQA (DET, L, ENTR) | Recurrence quantification analysis (3 indices) | — |
| LLE | Largest Lyapunov exponent (chaos indicator) | >0 (healthy) |

### Autonomic Indices (4)

| Biomarker | Description | Normal Range |
|-----------|-------------|--------------|
| PNS Index | Parasympathetic activity z-score | -2 to +2 |
| SNS Index | Sympathetic activity z-score | -2 to +2 |
| Baevsky Stress Index | Cardiovascular stress level | 50–150 |
| Autonomic Balance | PNS - SNS (positive = vagal dominant) | — |

### ECG Morphology (4)

| Biomarker | Description | Normal Range |
|-----------|-------------|--------------|
| QTc | Corrected QT interval | 360–440 ms |
| PR interval | Atrioventricular conduction time | 120–200 ms |
| QRS duration | Ventricular depolarization width | 80–120 ms |
| ST deviation | ST segment elevation/depression | -0.1–0.1 mV |

### PPG Vascular (5)

| Biomarker | Description | Normal Range |
|-----------|-------------|--------------|
| Augmentation Index | Arterial stiffness proxy | 0–30% |
| Respiratory Rate | Derived from PPG modulation | 12–20 rpm |
| Pulse Amplitude | Peak-to-trough amplitude | — |
| Rise Time | Systolic rise time | — ms |
| Pulse Transit Time | Estimated vascular transit | — ms |

### Arrhythmia (6)

HR class (normal / bradycardia / tachycardia), AFib suspicion, ectopic beat count, ectopic ratio, RR coefficient of variation, AFib evidence flags.

### Signal Quality (6)

Overall SQI score (0–100), SNR (dB), flatline fraction, clipping fraction, baseline wander fraction, quality label (Good / Acceptable / Poor / Unacceptable).

### Time-Frequency (3)

STFT spectrogram (LF/HF over time), CWT Morlet scalogram, dominant LF time percentage.

---

## AI Assistant (Step 6)

The AI Assistant tab provides a conversational interface to explore your analysis results using Google Gemini.

**Setup:**
1. Obtain a Google AI API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Enter the key in the left sidebar on the AI Assistant page
3. Click "Validate Key" — available models are fetched dynamically from the Gemini API
4. Select a model and start chatting

The assistant is automatically provided with the full analysis context (all computed biomarkers, risk levels, signal quality, arrhythmia findings) and responds according to HRV clinical guidelines (Task Force 1996, ESC/AHA, Shaffer & Ginsberg 2017).

> The assistant is for research and educational purposes only. Its responses do not constitute medical advice.

---

## API Reference

### POST /analysis/analyze

Runs full biomarker extraction on a signal.

Request body:
```json
{
  "signal": [0.1, 0.2, ...],
  "sampling_rate": 500,
  "signal_type": "ECG",
  "compute_advanced": true,
  "compute_autonomic": true,
  "compute_timefreq": false,
  "artifact_correction": true,
  "artifact_detection_method": "combined",
  "artifact_correction_method": "cubic_spline"
}
```

### GET /signals/sample/{signal_type}

Returns a sample ECG or PPG signal for testing. `signal_type` is `ECG` or `PPG`.

### GET /health

Returns `{"status": "ok"}` when the backend is running.

---

## Disclaimer

This platform is developed for **research and educational purposes only**. It is not a CE-marked or FDA-cleared medical device. Do not use for clinical diagnosis without qualified medical supervision.

---

## License

MIT License — see [LICENSE](LICENSE)

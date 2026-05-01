from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import tempfile, os
from core.signal_loader import SignalLoader
from models.signal import SignalType

router = APIRouter(prefix="/signals", tags=["signals"])

@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/upload")
async def upload_signal(
    file: UploadFile = File(...),
    signal_type: str = Form(...),
    sampling_rate: int = Form(...),
    channel_name: str = Form(default=""),
):
    """Carica file ECG/PPG in formato CSV, Excel, EDF o WFDB e ritorna il segnale come array JSON."""
    try:
        suffix = os.path.splitext(file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        try:
            if suffix in (".csv", ".txt"):
                signal, fs = SignalLoader.from_csv(tmp_path, signal_col=None, time_col=None, fs=sampling_rate)
            elif suffix in (".xlsx", ".xls"):
                signal, fs = SignalLoader.from_excel(tmp_path, signal_col=None, fs=sampling_rate)
            elif suffix == ".edf":
                signal, fs = SignalLoader.from_edf(tmp_path, channel_name=channel_name or None)
            elif suffix in (".dat", ".hea", ""):
                signal, fs = SignalLoader.from_wfdb(tmp_path.replace(suffix, ""), channel_idx=0)
            else:
                raise ValueError(f"Formato non supportato: {suffix}")
        finally:
            os.unlink(tmp_path)
        return {
            "values": signal.tolist(),
            "sampling_rate": fs,
            "n_samples": len(signal),
            "duration_seconds": len(signal) / fs,
            "signal_type": signal_type,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/sample/{signal_type}")
def get_sample(signal_type: str):
    """Ritorna un segnale campione sintetico ECG o PPG per demo."""
    import neurokit2 as nk
    if signal_type.upper() == "ECG":
        sig = nk.ecg_simulate(duration=30, sampling_rate=500, noise=0.02)
        fs = 500
    elif signal_type.upper() == "PPG":
        sig = nk.ppg_simulate(duration=30, sampling_rate=125, heart_rate=70)
        fs = 125
    else:
        raise HTTPException(status_code=400, detail="signal_type deve essere ECG o PPG")
    return {
        "values": sig.tolist(),
        "sampling_rate": fs,
        "n_samples": len(sig),
        "duration_seconds": 30.0,
        "signal_type": signal_type.upper(),
    }

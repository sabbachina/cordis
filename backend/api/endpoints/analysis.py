from fastapi import APIRouter, HTTPException
import numpy as np
from models.signal import AnalysisRequest, SignalType
from models.biomarker import BiomarkerReport
from core.signal_preprocessor import SignalPreprocessor
from core.biomarker_extractor import BiomarkerExtractor
from core.ml_classifier import AnomalyClassifier

router = APIRouter(prefix="/analysis", tags=["analysis"])
_classifier = AnomalyClassifier()

@router.post("/analyze", response_model=BiomarkerReport)
def analyze_signal(request: AnalysisRequest):
    """Pipeline completa: preprocessing → peak detection → biomarker extraction → ML."""
    try:
        signal = np.array(request.signal.values, dtype=float)
        fs = request.signal.sampling_rate
        warnings = []

        # Preprocessing
        signal_clean = SignalPreprocessor.preprocess(signal, fs, request.preprocessing)

        # Biomarker extraction
        extractor = BiomarkerExtractor()
        report = extractor.extract(
            signal=signal_clean,
            fs=fs,
            signal_type=request.signal.signal_type,
            config=request.preprocessing,
            compute_hrv=request.compute_hrv,
            compute_morphology=request.compute_morphology,
            warnings_list=warnings,
        )

        # ML anomaly detection
        if request.compute_ml:
            hrv_time = {}
            hrv_freq = {}
            morphology = {}
            if report.hrv_time:
                hrv_time = {
                    "mean_hr": report.hrv_time.mean_hr.value,
                    "sdnn": report.hrv_time.sdnn.value,
                    "rmssd": report.hrv_time.rmssd.value,
                    "pnn50": report.hrv_time.pnn50.value,
                }
            if report.hrv_freq:
                hrv_freq = {
                    "lf_hf_ratio": report.hrv_freq.lf_hf_ratio.value,
                    "lf_power": report.hrv_freq.lf_power.value,
                    "hf_power": report.hrv_freq.hf_power.value,
                }
            if report.ecg_morphology:
                morphology = {
                    "qtc": report.ecg_morphology.qtc_ms.value,
                    "st_deviation": report.ecg_morphology.st_deviation_mv.value,
                }
            report.ml_anomaly = _classifier.classify(hrv_time, hrv_freq, morphology)

        report.warnings.extend(warnings)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

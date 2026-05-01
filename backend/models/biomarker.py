from pydantic import BaseModel
from typing import Optional
from enum import Enum


class RiskLevel(str, Enum):
    NORMAL = "normal"
    BORDERLINE = "borderline"
    ABNORMAL = "abnormal"
    UNKNOWN = "unknown"


class BiomarkerValue(BaseModel):
    name: str
    value: Optional[float]
    unit: str
    normal_range: tuple[float, float]
    risk_level: RiskLevel
    description: str


class HRVTimeReport(BaseModel):
    mean_hr: BiomarkerValue
    sdnn: BiomarkerValue
    rmssd: BiomarkerValue
    pnn50: BiomarkerValue
    pnn20: BiomarkerValue


class HRVFreqReport(BaseModel):
    vlf_power: BiomarkerValue
    lf_power: BiomarkerValue
    hf_power: BiomarkerValue
    lf_hf_ratio: BiomarkerValue


class ECGMorphologyReport(BaseModel):
    pr_interval_ms: BiomarkerValue
    qrs_duration_ms: BiomarkerValue
    qtc_ms: BiomarkerValue
    st_deviation_mv: BiomarkerValue


class PPGVascularReport(BaseModel):
    pulse_amplitude: BiomarkerValue
    augmentation_index: BiomarkerValue
    respiratory_rate: BiomarkerValue


class MLAnomalyReport(BaseModel):
    anomaly_score: float
    is_anomalous: bool
    flags: list[str]
    confidence: float


class BiomarkerReport(BaseModel):
    signal_type: str
    duration_seconds: float
    sampling_rate: int
    n_peaks_detected: int
    hrv_time: Optional[HRVTimeReport] = None
    hrv_freq: Optional[HRVFreqReport] = None
    ecg_morphology: Optional[ECGMorphologyReport] = None
    ppg_vascular: Optional[PPGVascularReport] = None
    ml_anomaly: Optional[MLAnomalyReport] = None
    warnings: list[str] = []

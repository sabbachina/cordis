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
    nn50: Optional[BiomarkerValue] = None
    sdann: Optional[BiomarkerValue] = None
    sdnni: Optional[BiomarkerValue] = None
    hrvi: Optional[BiomarkerValue] = None
    tinn: Optional[BiomarkerValue] = None


class HRVFreqReport(BaseModel):
    vlf_power: BiomarkerValue
    lf_power: BiomarkerValue
    hf_power: BiomarkerValue
    lf_hf_ratio: BiomarkerValue
    total_power: Optional[BiomarkerValue] = None
    lfnu: Optional[BiomarkerValue] = None
    hfnu: Optional[BiomarkerValue] = None


class ECGMorphologyReport(BaseModel):
    pr_interval_ms: BiomarkerValue
    qrs_duration_ms: BiomarkerValue
    qtc_ms: BiomarkerValue
    st_deviation_mv: BiomarkerValue


class PPGVascularReport(BaseModel):
    pulse_amplitude: BiomarkerValue
    augmentation_index: BiomarkerValue
    stiffness_index: Optional[BiomarkerValue] = None
    reflection_index: Optional[BiomarkerValue] = None
    respiratory_rate: BiomarkerValue


class HRVNonlinearReport(BaseModel):
    sd1: BiomarkerValue          # Poincaré SD1
    sd2: BiomarkerValue          # Poincaré SD2
    sd1_sd2_ratio: BiomarkerValue
    sample_entropy: BiomarkerValue
    dfa_alpha1: BiomarkerValue   # DFA short-range scaling exponent


class HRVAdvancedReport(BaseModel):
    """Advanced nonlinear HRV — Kubios Scientific level."""
    dfa_alpha2: Optional[BiomarkerValue] = None
    approximate_entropy: Optional[BiomarkerValue] = None
    fuzzy_entropy: Optional[BiomarkerValue] = None
    mse_slope: Optional[BiomarkerValue] = None
    rqa_rr_pct: Optional[BiomarkerValue] = None
    rqa_det: Optional[BiomarkerValue] = None
    rqa_entr: Optional[BiomarkerValue] = None
    lyapunov_exponent: Optional[BiomarkerValue] = None


class AutonomicReport(BaseModel):
    pns_index: Optional[BiomarkerValue] = None
    sns_index: Optional[BiomarkerValue] = None
    baevsky_stress_index: Optional[BiomarkerValue] = None
    autonomic_balance: Optional[BiomarkerValue] = None


class ArrhythmiaReport(BaseModel):
    afib_suspected: bool
    afib_evidence: list[str]
    ectopic_beats: int
    ectopic_ratio: float
    heart_rate_class: str        # normal | bradycardia | tachycardia
    rr_cv: float


class SignalQualityReport(BaseModel):
    overall_score: float         # 0–100
    quality_label: str           # Good / Acceptable / Poor / Unacceptable
    snr_db: float
    flatline_fraction: float
    clipping_fraction: float
    baseline_wander_fraction: float
    hf_noise_fraction: float
    warnings: list[str]


class MLAnomalyReport(BaseModel):
    anomaly_score: float
    is_anomalous: bool
    flags: list[str]
    confidence: float


class TimeFreqReport(BaseModel):
    """STFT/CWT time-frequency analysis summary."""
    has_stft: bool = False
    has_cwt: bool = False
    lf_hf_variability: Optional[float] = None   # std of LF/HF over time (non-stationarity index)
    dominant_lf_time_pct: Optional[float] = None # % of time where LF > HF


class ArtifactReport(BaseModel):
    n_total_beats: int
    n_artifacts: int
    artifact_ratio: float
    artifact_pct: float
    quality_label: str
    correctable: bool


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
    hrv_nonlinear: Optional[HRVNonlinearReport] = None
    arrhythmia: Optional[ArrhythmiaReport] = None
    signal_quality: Optional[SignalQualityReport] = None
    time_freq: Optional[TimeFreqReport] = None
    artifact_correction: Optional[ArtifactReport] = None
    hrv_advanced: Optional[HRVAdvancedReport] = None
    autonomic: Optional[AutonomicReport] = None
    warnings: list[str] = []

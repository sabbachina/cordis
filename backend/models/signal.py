from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional


class SignalType(str, Enum):
    ECG = "ECG"
    PPG = "PPG"


class SignalInput(BaseModel):
    signal_type: SignalType
    sampling_rate: int = Field(..., ge=50, le=2000, description="Hz")
    values: list[float] = Field(..., min_length=100)
    time: Optional[list[float]] = None
    channel_name: Optional[str] = None


class PreprocessingConfig(BaseModel):
    lowcut: float = Field(default=0.5, ge=0.1, le=5.0)
    highcut: float = Field(default=40.0, ge=5.0, le=500.0)
    method: str = Field(default="butterworth", pattern="^(butterworth|savgol|neurokit)$")
    remove_baseline: bool = True


class AnalysisRequest(BaseModel):
    signal: SignalInput
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    compute_hrv: bool = True
    compute_morphology: bool = True
    compute_ml: bool = True
    compute_nonlinear: bool = True
    compute_arrhythmia: bool = True
    compute_signal_quality: bool = True
    compute_timefreq: bool = False    # Costoso, off by default
    artifact_correction: bool = True  # On by default
    artifact_detection_method: str = Field(default="combined", pattern="^(threshold|quotient|moving_median|combined)$")
    artifact_correction_method: str = Field(default="cubic_spline", pattern="^(linear|cubic_spline|moving_average|delete)$")
    compute_advanced: bool = True   # DFA α2, ApEn, FuzzyEn, MSE, RQA, LLE
    compute_autonomic: bool = True  # PNS, SNS, Baevsky

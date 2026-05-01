"""
BiomarkerExtractor — orchestrates ECG/PPG analysis pipelines and maps
raw feature dicts to fully typed BiomarkerReport Pydantic models.

Usage
-----
    extractor = BiomarkerExtractor()
    report = extractor.extract(signal, fs, signal_type, config,
                               compute_hrv=True, compute_morphology=True,
                               warnings_list=[])
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from models.signal import PreprocessingConfig, SignalType
from models.biomarker import (
    RiskLevel,
    BiomarkerValue,
    HRVTimeReport,
    HRVFreqReport,
    ECGMorphologyReport,
    PPGVascularReport,
    BiomarkerReport,
)
from core.ecg_analyzer import ECGAnalyzer
from core.ppg_analyzer import PPGAnalyzer


class BiomarkerExtractor:
    """
    Maps raw analysis dicts from ECGAnalyzer / PPGAnalyzer into the typed
    BiomarkerReport hierarchy.

    All clinical thresholds are derived from standard published guidelines
    and are hardcoded as class-level constants for transparency.
    """

    # ------------------------------------------------------------------
    # Clinical thresholds (hardcoded)
    # ------------------------------------------------------------------
    # Format: (normal_min, normal_max, borderline_low, borderline_high)
    # borderline_low  = lower edge of the borderline LOW zone  (< normal_min)
    # borderline_high = upper edge of the borderline HIGH zone (> normal_max)
    # Values outside the borderline range → ABNORMAL

    _THRESHOLDS: dict[str, tuple[float, float, float, float]] = {
        # HRV time-domain
        "sdnn":       (34.0,  66.0,  20.0,  90.0),   # ms
        "rmssd":      (27.0,  57.0,  15.0,  80.0),   # ms
        "pnn50":      (2.0,   10.0,  0.5,   25.0),   # %
        "mean_hr":    (60.0,  100.0, 50.0,  110.0),  # bpm
        # HRV frequency-domain
        "lf_hf_ratio":(1.0,   3.0,  0.5,   5.0),    # ratio
        # ECG morphology
        "qtc_ms":     (360.0, 440.0, 320.0, 500.0),  # ms  (>500 → ABNORMAL)
        "pr_interval_ms": (120.0, 200.0, 100.0, 220.0),  # ms
        "qrs_duration_ms":(80.0, 120.0,  60.0, 150.0),   # ms
        "st_deviation_mv":(-0.1, 0.1,   -0.2,  0.2),     # mV
        # PPG vascular
        "augmentation_index": (0.0, 30.0, -10.0, 50.0),  # %
        "respiratory_rate":  (12.0, 20.0,   8.0, 25.0),  # breaths/min
    }

    # Normal ranges displayed to the user (for BiomarkerValue.normal_range)
    _NORMAL_RANGES: dict[str, tuple[float, float]] = {
        k: (v[0], v[1]) for k, v in _THRESHOLDS.items()
    }

    # ------------------------------------------------------------------
    # Classification helper
    # ------------------------------------------------------------------
    @staticmethod
    def _classify(
        value: Optional[float],
        normal_min: float,
        normal_max: float,
        borderline_low: float,
        borderline_high: float,
    ) -> RiskLevel:
        """
        Assign a RiskLevel to *value* given four threshold boundaries.

        NORMAL     : normal_min  ≤ value ≤ normal_max
        BORDERLINE : borderline_low ≤ value < normal_min
                     OR normal_max < value ≤ borderline_high
        ABNORMAL   : value < borderline_low OR value > borderline_high
        UNKNOWN    : value is None or NaN
        """
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return RiskLevel.UNKNOWN

        if normal_min <= value <= normal_max:
            return RiskLevel.NORMAL
        if borderline_low <= value <= borderline_high:
            return RiskLevel.BORDERLINE
        return RiskLevel.ABNORMAL

    # ------------------------------------------------------------------
    # BiomarkerValue factory
    # ------------------------------------------------------------------
    def _make_biomarker(
        self,
        key: str,
        value: Optional[float],
        unit: str,
        description: str,
    ) -> BiomarkerValue:
        """
        Build a BiomarkerValue for *key* using the hardcoded threshold table.
        """
        thresholds = self._THRESHOLDS[key]
        normal_range = self._NORMAL_RANGES[key]
        risk_level = self._classify(value, *thresholds)

        # Round to 3 decimal places for cleanliness (keep None as-is)
        display_value = round(float(value), 3) if value is not None and not np.isnan(value) else None

        return BiomarkerValue(
            name=key,
            value=display_value,
            unit=unit,
            normal_range=normal_range,
            risk_level=risk_level,
            description=description,
        )

    # ------------------------------------------------------------------
    # HRV time-domain sub-report
    # ------------------------------------------------------------------
    def _build_hrv_time(self, hrv_time_dict: dict) -> HRVTimeReport:
        return HRVTimeReport(
            mean_hr=self._make_biomarker(
                "mean_hr",
                hrv_time_dict.get("mean_hr"),
                "bpm",
                "Mean heart rate derived from RR intervals.",
            ),
            sdnn=self._make_biomarker(
                "sdnn",
                hrv_time_dict.get("sdnn"),
                "ms",
                "Standard deviation of all NN intervals — overall HRV.",
            ),
            rmssd=self._make_biomarker(
                "rmssd",
                hrv_time_dict.get("rmssd"),
                "ms",
                "Root mean square of successive differences — short-term vagal activity.",
            ),
            pnn50=self._make_biomarker(
                "pnn50",
                hrv_time_dict.get("pnn50"),
                "%",
                "Percentage of successive RR differences > 50 ms.",
            ),
            pnn20=BiomarkerValue(
                name="pnn20",
                value=round(float(hrv_time_dict["pnn20"]), 3)
                      if hrv_time_dict.get("pnn20") is not None
                         and not np.isnan(float(hrv_time_dict["pnn20"]))
                      else None,
                unit="%",
                normal_range=(5.0, 30.0),
                risk_level=RiskLevel.UNKNOWN,   # No specific thresholds — informational
                description="Percentage of successive RR differences > 20 ms.",
            ),
        )

    # ------------------------------------------------------------------
    # HRV frequency-domain sub-report
    # ------------------------------------------------------------------
    def _build_hrv_freq(self, hrv_freq_dict: dict) -> HRVFreqReport:
        def _bv_power(key: str, desc: str) -> BiomarkerValue:
            v = hrv_freq_dict.get(key)
            return BiomarkerValue(
                name=key,
                value=round(float(v), 4) if v is not None and not np.isnan(float(v)) else None,
                unit="ms²",
                normal_range=(0.0, float("inf")),
                risk_level=RiskLevel.UNKNOWN,  # informational
                description=desc,
            )

        return HRVFreqReport(
            vlf_power=_bv_power("vlf_power", "Very-low-frequency HRV power (0.003–0.04 Hz)."),
            lf_power=_bv_power("lf_power",   "Low-frequency HRV power (0.04–0.15 Hz) — sympatho-vagal balance."),
            hf_power=_bv_power("hf_power",   "High-frequency HRV power (0.15–0.40 Hz) — parasympathetic activity."),
            lf_hf_ratio=self._make_biomarker(
                "lf_hf_ratio",
                hrv_freq_dict.get("lf_hf_ratio"),
                "ratio",
                "LF/HF ratio — index of sympatho-vagal balance.",
            ),
        )

    # ------------------------------------------------------------------
    # ECG morphology sub-report
    # ------------------------------------------------------------------
    def _build_ecg_morphology(self, morphology_dict: dict) -> ECGMorphologyReport:
        return ECGMorphologyReport(
            pr_interval_ms=self._make_biomarker(
                "pr_interval_ms",
                morphology_dict.get("pr_interval_ms"),
                "ms",
                "PR interval (P-wave onset to QRS onset) — AV conduction time.",
            ),
            qrs_duration_ms=self._make_biomarker(
                "qrs_duration_ms",
                morphology_dict.get("qrs_duration_ms"),
                "ms",
                "QRS complex duration — ventricular depolarization time.",
            ),
            qtc_ms=self._make_biomarker(
                "qtc_ms",
                morphology_dict.get("qtc_ms"),
                "ms",
                "Heart-rate corrected QT interval (Bazett formula) — repolarization.",
            ),
            st_deviation_mv=self._make_biomarker(
                "st_deviation_mv",
                morphology_dict.get("st_deviation_mv"),
                "mV",
                "Mean ST-segment deviation from isoelectric baseline — ischemia marker.",
            ),
        )

    # ------------------------------------------------------------------
    # PPG vascular sub-report
    # ------------------------------------------------------------------
    def _build_ppg_vascular(self, vascular_dict: dict) -> PPGVascularReport:
        # pulse_amplitude: informational — no fixed clinical thresholds
        pa = vascular_dict.get("pulse_amplitude")
        pulse_amp_bv = BiomarkerValue(
            name="pulse_amplitude",
            value=round(float(pa), 4) if pa is not None and not np.isnan(float(pa)) else None,
            unit="a.u.",
            normal_range=(0.5, 3.0),
            risk_level=RiskLevel.UNKNOWN,
            description="Mean systolic peak-to-trough amplitude (normalized signal).",
        )

        return PPGVascularReport(
            pulse_amplitude=pulse_amp_bv,
            augmentation_index=self._make_biomarker(
                "augmentation_index",
                vascular_dict.get("augmentation_index"),
                "%",
                "Augmentation Index (AIx) — arterial stiffness proxy.",
            ),
            respiratory_rate=self._make_biomarker(
                "respiratory_rate",
                vascular_dict.get("respiratory_rate"),
                "breaths/min",
                "Respiratory rate estimated from PPG amplitude modulation.",
            ),
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def extract(
        self,
        signal: np.ndarray,
        fs: int,
        signal_type: SignalType,
        config: PreprocessingConfig,
        compute_hrv: bool = True,
        compute_morphology: bool = True,
        warnings_list: Optional[list[str]] = None,
    ) -> BiomarkerReport:
        """
        Run the full analysis pipeline and return a BiomarkerReport.

        Parameters
        ----------
        signal            : preprocessed 1-D signal array (float64)
        fs                : sampling rate in Hz
        signal_type       : SignalType.ECG or SignalType.PPG
        config            : PreprocessingConfig (forwarded to analyzer)
        compute_hrv       : whether to compute HRV biomarkers
        compute_morphology: whether to compute ECG morphology or PPG vascular indices
        warnings_list     : external list to which analysis warnings are appended

        Returns
        -------
        BiomarkerReport
        """
        if warnings_list is None:
            warnings_list = []

        duration_seconds = len(signal) / fs

        # ----------------------------------------------------------------
        # Branch on signal type
        # ----------------------------------------------------------------
        if signal_type == SignalType.ECG:
            raw = ECGAnalyzer.analyze(
                signal=signal,
                fs=fs,
                config=config,
                compute_hrv=compute_hrv,
                compute_morphology=compute_morphology,
            )

            # Propagate analyzer warnings
            warnings_list.extend(raw.get("warnings", []))

            n_peaks = raw.get("n_peaks", 0)
            hrv_time_report: Optional[HRVTimeReport] = None
            hrv_freq_report: Optional[HRVFreqReport] = None
            ecg_morph_report: Optional[ECGMorphologyReport] = None

            if compute_hrv:
                if raw.get("hrv_time"):
                    hrv_time_report = self._build_hrv_time(raw["hrv_time"])
                if raw.get("hrv_freq"):
                    hrv_freq_report = self._build_hrv_freq(raw["hrv_freq"])

            if compute_morphology and raw.get("morphology"):
                ecg_morph_report = self._build_ecg_morphology(raw["morphology"])

            return BiomarkerReport(
                signal_type=SignalType.ECG,
                duration_seconds=round(duration_seconds, 3),
                sampling_rate=fs,
                n_peaks_detected=n_peaks,
                hrv_time=hrv_time_report,
                hrv_freq=hrv_freq_report,
                ecg_morphology=ecg_morph_report,
                ppg_vascular=None,
                ml_anomaly=None,
                warnings=list(warnings_list),
            )

        elif signal_type == SignalType.PPG:
            raw = PPGAnalyzer.analyze(
                signal=signal,
                fs=fs,
                config=config,
                compute_hrv=compute_hrv,
                compute_vascular=compute_morphology,  # reuse flag
            )

            warnings_list.extend(raw.get("warnings", []))

            n_peaks = raw.get("n_peaks", 0)
            hrv_time_report = None
            hrv_freq_report = None
            ppg_vasc_report: Optional[PPGVascularReport] = None

            if compute_hrv:
                if raw.get("hrv_time"):
                    hrv_time_report = self._build_hrv_time(raw["hrv_time"])
                if raw.get("hrv_freq"):
                    hrv_freq_report = self._build_hrv_freq(raw["hrv_freq"])

            if compute_morphology and raw.get("vascular"):
                ppg_vasc_report = self._build_ppg_vascular(raw["vascular"])

            return BiomarkerReport(
                signal_type=SignalType.PPG,
                duration_seconds=round(duration_seconds, 3),
                sampling_rate=fs,
                n_peaks_detected=n_peaks,
                hrv_time=hrv_time_report,
                hrv_freq=hrv_freq_report,
                ecg_morphology=None,
                ppg_vascular=ppg_vasc_report,
                ml_anomaly=None,
                warnings=list(warnings_list),
            )

        else:
            raise ValueError(f"Unsupported signal_type: {signal_type!r}")

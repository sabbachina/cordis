"""
BiomarkerExtractor — orchestrates ECG/PPG analysis pipelines and maps
raw feature dicts to fully typed BiomarkerReport Pydantic models.

Usage
-----
    extractor = BiomarkerExtractor()
    report = extractor.extract(signal, fs, signal_type, config,
                               compute_hrv=True, compute_morphology=True,
                               compute_nonlinear=True, compute_arrhythmia=True,
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
    HRVNonlinearReport,
    ArrhythmiaReport,
    SignalQualityReport,
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
    # BiomarkerValue factory — key-based (uses hardcoded threshold table)
    # ------------------------------------------------------------------
    def _make_biomarker_from_key(
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
        display_value = (
            round(float(value), 3)
            if value is not None and not (isinstance(value, float) and np.isnan(value))
            else None
        )

        return BiomarkerValue(
            name=key,
            value=display_value,
            unit=unit,
            normal_range=normal_range,
            risk_level=risk_level,
            description=description,
        )

    # ------------------------------------------------------------------
    # BiomarkerValue factory — explicit thresholds (for new biomarkers)
    # ------------------------------------------------------------------
    @staticmethod
    def _make_biomarker(
        name: str,
        value: Optional[float],
        unit: str,
        normal_range: tuple,
        thresholds: tuple,
        description: str,
    ) -> BiomarkerValue:
        """
        Build a BiomarkerValue with explicitly supplied thresholds.

        Parameters
        ----------
        name         : display name
        value        : raw float (may be None / NaN)
        unit         : physical unit string
        normal_range : (normal_min, normal_max)
        thresholds   : (normal_min, normal_max, borderline_low, borderline_high)
        description  : clinical description
        """
        display_value = (
            round(float(value), 4)
            if value is not None and not (isinstance(value, float) and np.isnan(value))
            else None
        )
        risk_level = BiomarkerExtractor._classify(value, *thresholds)
        return BiomarkerValue(
            name=name,
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
            mean_hr=self._make_biomarker_from_key(
                "mean_hr",
                hrv_time_dict.get("mean_hr"),
                "bpm",
                "Mean heart rate derived from RR intervals.",
            ),
            sdnn=self._make_biomarker_from_key(
                "sdnn",
                hrv_time_dict.get("sdnn"),
                "ms",
                "Standard deviation of all NN intervals — overall HRV.",
            ),
            rmssd=self._make_biomarker_from_key(
                "rmssd",
                hrv_time_dict.get("rmssd"),
                "ms",
                "Root mean square of successive differences — short-term vagal activity.",
            ),
            pnn50=self._make_biomarker_from_key(
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
            lf_hf_ratio=self._make_biomarker_from_key(
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
            pr_interval_ms=self._make_biomarker_from_key(
                "pr_interval_ms",
                morphology_dict.get("pr_interval_ms"),
                "ms",
                "PR interval (P-wave onset to QRS onset) — AV conduction time.",
            ),
            qrs_duration_ms=self._make_biomarker_from_key(
                "qrs_duration_ms",
                morphology_dict.get("qrs_duration_ms"),
                "ms",
                "QRS complex duration — ventricular depolarization time.",
            ),
            qtc_ms=self._make_biomarker_from_key(
                "qtc_ms",
                morphology_dict.get("qtc_ms"),
                "ms",
                "Heart-rate corrected QT interval (Bazett formula) — repolarization.",
            ),
            st_deviation_mv=self._make_biomarker_from_key(
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

        # stiffness_index: informational
        si = vascular_dict.get("stiffness_index")
        si_bv: Optional[BiomarkerValue] = None
        if si is not None:
            si_bv = BiomarkerValue(
                name="stiffness_index",
                value=round(float(si), 4) if not np.isnan(float(si)) else None,
                unit="a.u.",
                normal_range=(5.0, 10.0),
                risk_level=RiskLevel.UNKNOWN,
                description="Stiffness Index — proxy for arterial wall stiffness.",
            )

        # reflection_index
        ri = vascular_dict.get("reflection_index")
        ri_bv: Optional[BiomarkerValue] = None
        if ri is not None:
            ri_bv = self._make_biomarker(
                "Reflection Index",
                ri,
                "%",
                (0.0, 50.0),
                (0.0, 50.0, -5.0, 80.0),
                "Reflection Index (RI) — dicrotic notch depth relative to systolic amplitude.",
            )

        return PPGVascularReport(
            pulse_amplitude=pulse_amp_bv,
            augmentation_index=self._make_biomarker_from_key(
                "augmentation_index",
                vascular_dict.get("augmentation_index"),
                "%",
                "Augmentation Index (AIx) — arterial stiffness proxy.",
            ),
            stiffness_index=si_bv,
            reflection_index=ri_bv,
            respiratory_rate=self._make_biomarker_from_key(
                "respiratory_rate",
                vascular_dict.get("respiratory_rate"),
                "breaths/min",
                "Respiratory rate estimated from PPG amplitude modulation.",
            ),
        )

    # ------------------------------------------------------------------
    # HRV non-linear sub-report
    # ------------------------------------------------------------------
    def _build_hrv_nonlinear(self, nl: dict) -> HRVNonlinearReport:
        return HRVNonlinearReport(
            sd1=self._make_biomarker(
                "SD1 (Poincaré)", nl.get("sd1"), "ms",
                (10.0, 50.0), (5.0, 10.0, 50.0, 80.0),
                "Short-term HRV variability (Poincaré SD1).",
            ),
            sd2=self._make_biomarker(
                "SD2 (Poincaré)", nl.get("sd2"), "ms",
                (20.0, 80.0), (10.0, 20.0, 80.0, 120.0),
                "Long-term HRV variability (Poincaré SD2).",
            ),
            sd1_sd2_ratio=self._make_biomarker(
                "SD1/SD2 Ratio", nl.get("sd1_sd2_ratio"), "ratio",
                (0.5, 1.5), (0.3, 0.5, 1.5, 2.5),
                "Autonomic balance — ratio of short-term to long-term variability.",
            ),
            sample_entropy=self._make_biomarker(
                "Sample Entropy", nl.get("sample_entropy"), "a.u.",
                (1.0, 2.5), (0.5, 1.0, 2.5, 3.5),
                "Signal complexity / unpredictability of RR intervals.",
            ),
            dfa_alpha1=self._make_biomarker(
                "DFA α1", nl.get("dfa_alpha1"), "a.u.",
                (0.75, 1.25), (0.5, 0.75, 1.25, 1.5),
                "Short-range fractal correlation (Detrended Fluctuation Analysis).",
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
        compute_nonlinear: bool = True,
        compute_arrhythmia: bool = True,
        warnings_list: Optional[list[str]] = None,
    ) -> BiomarkerReport:
        """
        Run the full analysis pipeline and return a BiomarkerReport.

        Parameters
        ----------
        signal             : preprocessed 1-D signal array (float64)
        fs                 : sampling rate in Hz
        signal_type        : SignalType.ECG or SignalType.PPG
        config             : PreprocessingConfig (forwarded to analyzer)
        compute_hrv        : whether to compute HRV biomarkers
        compute_morphology : whether to compute ECG morphology or PPG vascular indices
        compute_nonlinear  : whether to compute non-linear HRV (ECG only)
        compute_arrhythmia : whether to run arrhythmia screening (ECG only)
        warnings_list      : external list to which analysis warnings are appended

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
                compute_nonlinear=compute_nonlinear,
                compute_arrhythmia=compute_arrhythmia,
            )

            # Propagate analyzer warnings
            warnings_list.extend(raw.get("warnings", []))

            n_peaks = raw.get("n_peaks", 0)
            hrv_time_report: Optional[HRVTimeReport] = None
            hrv_freq_report: Optional[HRVFreqReport] = None
            ecg_morph_report: Optional[ECGMorphologyReport] = None
            hrv_nl_report: Optional[HRVNonlinearReport] = None
            arrhythmia_report: Optional[ArrhythmiaReport] = None

            if compute_hrv:
                if raw.get("hrv_time"):
                    hrv_time_report = self._build_hrv_time(raw["hrv_time"])
                if raw.get("hrv_freq"):
                    hrv_freq_report = self._build_hrv_freq(raw["hrv_freq"])

            if compute_morphology and raw.get("morphology"):
                ecg_morph_report = self._build_ecg_morphology(raw["morphology"])

            if compute_nonlinear and raw.get("hrv_nonlinear"):
                hrv_nl_report = self._build_hrv_nonlinear(raw["hrv_nonlinear"])

            if compute_arrhythmia and raw.get("arrhythmia"):
                arr = raw["arrhythmia"]
                arrhythmia_report = ArrhythmiaReport(
                    afib_suspected=arr["afib_suspected"],
                    afib_evidence=arr["afib_evidence"],
                    ectopic_beats=arr["ectopic_beats"],
                    ectopic_ratio=arr["ectopic_ratio"],
                    heart_rate_class=arr["heart_rate_class"],
                    rr_cv=arr["rr_cv"],
                )

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
                hrv_nonlinear=hrv_nl_report,
                arrhythmia=arrhythmia_report,
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

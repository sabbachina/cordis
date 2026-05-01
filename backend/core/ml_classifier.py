"""
AnomalyClassifier — unsupervised anomaly detection for cardiac signals
using IsolationForest pre-fitted on synthetic normal HRV data, combined
with deterministic clinical rule checks.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from models.biomarker import MLAnomalyReport


class AnomalyClassifier:
    """
    Hybrid anomaly detector:
    1. IsolationForest trained on synthetic normal-range HRV data.
    2. Deterministic clinical flag rules applied on top.

    The IsolationForest is fitted once during __init__ via
    fit_on_normal_ranges() so the classifier is ready to use immediately.
    """

    # Feature order kept consistent across all methods
    FEATURE_NAMES = [
        "mean_hr",    # bpm
        "sdnn",       # ms
        "rmssd",      # ms
        "pnn50",      # %
        "lf_hf_ratio",# ratio
        "lf_power",   # ms²
        "hf_power",   # ms²
    ]

    # Synthetic normal ranges used to generate training data
    # (mean, std) for each feature
    _SYNTH_PARAMS: dict[str, tuple[float, float]] = {
        "mean_hr":     (72.0,  8.0),
        "sdnn":        (50.0,  12.0),
        "rmssd":       (42.0,  12.0),
        "pnn50":       (6.0,   3.0),
        "lf_hf_ratio": (2.0,   0.6),
        "lf_power":    (500.0, 150.0),
        "hf_power":    (350.0, 100.0),
    }

    # Clip bounds to keep synthetic data physiologically plausible
    _SYNTH_CLIPS: dict[str, tuple[float, float]] = {
        "mean_hr":     (45.0,  120.0),
        "sdnn":        (10.0,  150.0),
        "rmssd":       (5.0,   120.0),
        "pnn50":       (0.0,   40.0),
        "lf_hf_ratio": (0.1,   8.0),
        "lf_power":    (10.0,  2000.0),
        "hf_power":    (10.0,  2000.0),
    }

    def __init__(self, contamination: float = 0.1, random_state: int = 42) -> None:
        self._model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
        )
        self.fit_on_normal_ranges(n_samples=500, random_state=random_state)

    # ------------------------------------------------------------------
    # Synthetic training data + fit
    # ------------------------------------------------------------------
    def fit_on_normal_ranges(self, n_samples: int = 500, random_state: int = 42) -> None:
        """
        Generate *n_samples* synthetic normal-range feature vectors and fit
        the IsolationForest on them.

        Each feature is sampled from a Gaussian centred at its typical
        normal value and clipped to a physiologically plausible range.
        """
        rng = np.random.default_rng(seed=random_state)
        X = np.zeros((n_samples, len(self.FEATURE_NAMES)), dtype=np.float64)

        for col_idx, name in enumerate(self.FEATURE_NAMES):
            mean, std = self._SYNTH_PARAMS[name]
            lo, hi = self._SYNTH_CLIPS[name]
            samples = rng.normal(loc=mean, scale=std, size=n_samples)
            X[:, col_idx] = np.clip(samples, lo, hi)

        self._model.fit(X)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    @staticmethod
    def extract_features(
        hrv_time_dict: dict,
        hrv_freq_dict: dict,
    ) -> np.ndarray:
        """
        Pack HRV dicts into a 1-D feature vector of shape (7,).

        Missing or NaN values are replaced with the synthetic normal mean
        so the model can still produce a score.

        Parameters
        ----------
        hrv_time_dict : keys — mean_hr, sdnn, rmssd, pnn50 (+ pnn20)
        hrv_freq_dict : keys — lf_hf_ratio, lf_power, hf_power (+ vlf_power)

        Returns
        -------
        np.ndarray of shape (7,)
        """
        _defaults: dict[str, float] = {
            "mean_hr":     72.0,
            "sdnn":        50.0,
            "rmssd":       42.0,
            "pnn50":       6.0,
            "lf_hf_ratio": 2.0,
            "lf_power":    500.0,
            "hf_power":    350.0,
        }

        def _get(d: dict, key: str) -> float:
            v = d.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return _defaults[key]
            return float(v)

        features = np.array([
            _get(hrv_time_dict, "mean_hr"),
            _get(hrv_time_dict, "sdnn"),
            _get(hrv_time_dict, "rmssd"),
            _get(hrv_time_dict, "pnn50"),
            _get(hrv_freq_dict, "lf_hf_ratio"),
            _get(hrv_freq_dict, "lf_power"),
            _get(hrv_freq_dict, "hf_power"),
        ], dtype=np.float64)

        return features

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, features: np.ndarray) -> tuple[bool, float, float]:
        """
        Run the IsolationForest on *features*.

        Returns
        -------
        (is_anomalous, anomaly_score, confidence)
        - is_anomalous : True if IsolationForest labels as outlier (-1)
        - anomaly_score: raw decision function value (more negative → more anomalous)
        - confidence   : heuristic in [0, 1] derived from the score magnitude
        """
        X = features.reshape(1, -1)
        label = self._model.predict(X)[0]           # +1 normal, -1 anomaly
        raw_score = float(self._model.decision_function(X)[0])

        is_anomalous = bool(label == -1)

        # Map decision function to [0, 1] confidence:
        # typical range is roughly [-0.5, +0.5]; sigmoid-like transform
        confidence = float(1.0 / (1.0 + np.exp(5.0 * raw_score)))

        # Clamp to [0, 1]
        confidence = float(np.clip(confidence, 0.0, 1.0))

        # Normalise anomaly_score to [0, 1] (0 = normal, 1 = very anomalous)
        # using the same sigmoid on the negated raw score
        anomaly_score = float(np.clip(1.0 / (1.0 + np.exp(5.0 * raw_score)), 0.0, 1.0))

        return is_anomalous, anomaly_score, confidence

    # ------------------------------------------------------------------
    # Clinical rule engine
    # ------------------------------------------------------------------
    @staticmethod
    def apply_clinical_rules(
        hrv_time_dict: dict,
        hrv_freq_dict: dict,
        morphology_dict: dict,
    ) -> list[str]:
        """
        Apply deterministic clinical threshold rules and return a list of
        human-readable flag strings.

        Parameters
        ----------
        hrv_time_dict  : may contain mean_hr, sdnn, rmssd, pnn50
        hrv_freq_dict  : may contain lf_hf_ratio
        morphology_dict: may contain qtc, st_deviation (ECG) or other keys

        Returns
        -------
        list[str] — empty list when no rules fire
        """
        flags: list[str] = []

        def _val(d: dict, key: str, fallback: float = float("nan")) -> float:
            v = d.get(key)
            return float(v) if v is not None else fallback

        mean_hr = _val(hrv_time_dict, "mean_hr")
        rmssd   = _val(hrv_time_dict, "rmssd")
        lf_hf   = _val(hrv_freq_dict, "lf_hf_ratio")
        qtc     = _val(morphology_dict, "qtc")
        st_dev  = _val(morphology_dict, "st_deviation")

        if not np.isnan(qtc) and qtc > 440:
            flags.append("QTc prolongato (>440ms)")

        if not np.isnan(mean_hr) and mean_hr < 50:
            flags.append("Bradicardia (HR < 50 bpm)")

        if not np.isnan(mean_hr) and mean_hr > 100:
            flags.append("Tachicardia (HR > 100 bpm)")

        if not np.isnan(rmssd) and rmssd < 20:
            flags.append("RMSSD basso - ridotta variabilità (< 20ms)")

        if not np.isnan(lf_hf) and lf_hf > 4.0:
            flags.append("LF/HF elevato - predominanza simpatica")

        if not np.isnan(st_dev) and abs(st_dev) > 0.2:
            flags.append("Deviazione ST significativa")

        return flags

    # ------------------------------------------------------------------
    # High-level classify
    # ------------------------------------------------------------------
    def classify(
        self,
        hrv_time_dict: dict,
        hrv_freq_dict: dict,
        morphology_dict: dict,
    ) -> MLAnomalyReport:
        """
        Full classification: IsolationForest + clinical rules.

        Parameters
        ----------
        hrv_time_dict  : HRV time-domain values (keys: mean_hr, sdnn, rmssd, pnn50)
        hrv_freq_dict  : HRV frequency values (keys: lf_hf_ratio, lf_power, hf_power)
        morphology_dict: morphology values (keys: qtc, st_deviation)

        Returns
        -------
        MLAnomalyReport
        """
        features = self.extract_features(hrv_time_dict, hrv_freq_dict)
        is_anomalous, anomaly_score, confidence = self.predict(features)

        flags = self.apply_clinical_rules(hrv_time_dict, hrv_freq_dict, morphology_dict)

        # If any clinical rule fires, override ML label
        if flags:
            is_anomalous = True

        return MLAnomalyReport(
            anomaly_score=round(anomaly_score, 4),
            is_anomalous=is_anomalous,
            flags=flags,
            confidence=round(confidence, 4),
        )

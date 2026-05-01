"""
Test suite for ECGAnalyzer — R-peak detection, HRV time/freq domain,
morphology, end-to-end analyze(), and edge cases.
"""
import sys
import os
import warnings

import numpy as np
import pytest

# sys.path already patched by conftest.py
from core.ecg_analyzer import ECGAnalyzer
from models.signal import PreprocessingConfig


# ---------------------------------------------------------------------------
# Test 1 – detect_peaks
# ---------------------------------------------------------------------------

class TestDetectPeaks:
    def test_returns_nonempty_array(self, ecg_signal):
        signal, fs = ecg_signal
        peaks = ECGAnalyzer.detect_peaks(signal, fs)
        assert isinstance(peaks, np.ndarray), "Should return np.ndarray"
        assert len(peaks) > 0, "Peaks array should not be empty"

    def test_at_least_20_peaks_in_30s(self, ecg_signal):
        """30-second ECG at ~60–100 bpm should yield >= 20 R-peaks."""
        signal, fs = ecg_signal
        peaks = ECGAnalyzer.detect_peaks(signal, fs)
        assert len(peaks) >= 20, (
            f"Expected >= 20 R-peaks in 30 s ECG, got {len(peaks)}"
        )

    def test_peaks_are_within_signal_bounds(self, ecg_signal):
        signal, fs = ecg_signal
        peaks = ECGAnalyzer.detect_peaks(signal, fs)
        assert np.all(peaks >= 0)
        assert np.all(peaks < len(signal))

    def test_peaks_dtype_is_integer(self, ecg_signal):
        signal, fs = ecg_signal
        peaks = ECGAnalyzer.detect_peaks(signal, fs)
        assert np.issubdtype(peaks.dtype, np.integer), (
            f"Expected integer dtype, got {peaks.dtype}"
        )

    def test_peaks_are_sorted_ascending(self, ecg_signal):
        signal, fs = ecg_signal
        peaks = ECGAnalyzer.detect_peaks(signal, fs)
        assert np.all(np.diff(peaks) > 0), "Peaks must be in ascending order"


# ---------------------------------------------------------------------------
# Test 2 – compute_hrv_time
# ---------------------------------------------------------------------------

class TestComputeHRVTime:
    def test_constant_rr_sdnn_near_zero(self):
        """Perfectly regular rhythm: SDNN and RMSSD must be ~0."""
        rr = np.array([800.0] * 60)
        hrv = ECGAnalyzer.compute_hrv_time(rr)
        assert hrv["sdnn"] == pytest.approx(0.0, abs=1e-6), (
            f"SDNN should be 0 for constant RR, got {hrv['sdnn']}"
        )

    def test_constant_rr_rmssd_near_zero(self):
        rr = np.array([800.0] * 60)
        hrv = ECGAnalyzer.compute_hrv_time(rr)
        assert hrv["rmssd"] == pytest.approx(0.0, abs=1e-6), (
            f"RMSSD should be 0 for constant RR, got {hrv['rmssd']}"
        )

    def test_constant_rr_mean_hr_is_75(self):
        """800 ms RR interval → 60000/800 = 75 bpm."""
        rr = np.array([800.0] * 60)
        hrv = ECGAnalyzer.compute_hrv_time(rr)
        assert hrv["mean_hr"] == pytest.approx(75.0, rel=1e-3), (
            f"Expected mean_HR 75 bpm, got {hrv['mean_hr']}"
        )

    def test_returns_required_keys(self):
        rr = np.array([800.0] * 30)
        hrv = ECGAnalyzer.compute_hrv_time(rr)
        for key in ("mean_hr", "sdnn", "rmssd", "pnn50", "pnn20"):
            assert key in hrv, f"Missing key: {key}"

    def test_pnn50_zero_for_constant_rr(self):
        rr = np.array([800.0] * 60)
        hrv = ECGAnalyzer.compute_hrv_time(rr)
        assert hrv["pnn50"] == pytest.approx(0.0, abs=1e-6)

    def test_variable_rr_nonzero_sdnn(self):
        """Alternating short/long intervals → non-zero SDNN."""
        rr = np.array([700.0, 900.0] * 30)
        hrv = ECGAnalyzer.compute_hrv_time(rr)
        assert hrv["sdnn"] > 0

    def test_too_few_intervals_raises(self):
        with pytest.raises(ValueError):
            ECGAnalyzer.compute_hrv_time(np.array([800.0]))

    def test_values_are_finite(self):
        rr = np.linspace(700, 900, 50)
        hrv = ECGAnalyzer.compute_hrv_time(rr)
        for key in ("sdnn", "rmssd", "pnn50", "pnn20"):
            assert np.isfinite(hrv[key]), f"{key} should be finite"


# ---------------------------------------------------------------------------
# Test 3 – compute_hrv_freq
# ---------------------------------------------------------------------------

class TestComputeHRVFreq:
    def _normal_rr(self, n: int = 120) -> np.ndarray:
        """Simulate ~75 bpm with mild HRV."""
        rng = np.random.default_rng(42)
        return 800.0 + rng.normal(0, 30, n)

    def test_returns_required_keys(self):
        rr = self._normal_rr()
        freq = ECGAnalyzer.compute_hrv_freq(rr)
        for key in ("vlf_power", "lf_power", "hf_power", "lf_hf_ratio"):
            assert key in freq, f"Missing key: {key}"

    def test_power_values_are_non_negative(self):
        rr = self._normal_rr()
        freq = ECGAnalyzer.compute_hrv_freq(rr)
        for key in ("vlf_power", "lf_power", "hf_power"):
            assert freq[key] >= 0.0, f"{key} must be non-negative"

    def test_lf_hf_ratio_is_numeric(self):
        rr = self._normal_rr()
        freq = ECGAnalyzer.compute_hrv_freq(rr)
        assert isinstance(freq["lf_hf_ratio"], float)

    def test_too_few_rr_raises(self):
        with pytest.raises(ValueError):
            ECGAnalyzer.compute_hrv_freq(np.array([800.0, 810.0, 790.0]))

    def test_lf_hf_ratio_positive_for_normal_rr(self):
        rr = self._normal_rr()
        freq = ECGAnalyzer.compute_hrv_freq(rr)
        # lf_hf_ratio should be positive for typical HRV
        if not np.isnan(freq["lf_hf_ratio"]):
            assert freq["lf_hf_ratio"] > 0


# ---------------------------------------------------------------------------
# Test 4 – analyze() end-to-end
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_returns_dict_with_required_keys(self, ecg_signal):
        signal, fs = ecg_signal
        config = PreprocessingConfig()
        result = ECGAnalyzer.analyze(signal, fs, config)
        for key in ("n_peaks", "rr_intervals_ms", "hrv_time", "hrv_freq",
                    "morphology", "warnings"):
            assert key in result, f"Missing key in analyze() output: {key}"

    def test_n_peaks_is_positive(self, ecg_signal):
        signal, fs = ecg_signal
        config = PreprocessingConfig()
        result = ECGAnalyzer.analyze(signal, fs, config)
        assert result["n_peaks"] > 0, "Should detect at least one R-peak"

    def test_rr_intervals_are_list(self, ecg_signal):
        signal, fs = ecg_signal
        config = PreprocessingConfig()
        result = ECGAnalyzer.analyze(signal, fs, config)
        assert isinstance(result["rr_intervals_ms"], list)

    def test_hrv_time_is_dict_or_none(self, ecg_signal):
        signal, fs = ecg_signal
        config = PreprocessingConfig()
        result = ECGAnalyzer.analyze(signal, fs, config)
        assert result["hrv_time"] is None or isinstance(result["hrv_time"], dict)

    def test_hrv_time_contains_mean_hr_when_present(self, ecg_signal):
        signal, fs = ecg_signal
        config = PreprocessingConfig()
        result = ECGAnalyzer.analyze(signal, fs, config)
        if result["hrv_time"] is not None:
            assert "mean_hr" in result["hrv_time"]
            assert np.isfinite(result["hrv_time"]["mean_hr"])

    def test_hrv_freq_contains_lf_hf_when_present(self, ecg_signal):
        signal, fs = ecg_signal
        config = PreprocessingConfig()
        result = ECGAnalyzer.analyze(signal, fs, config)
        if result["hrv_freq"] is not None:
            assert "lf_power" in result["hrv_freq"]
            assert "hf_power" in result["hrv_freq"]

    def test_warnings_is_list(self, ecg_signal):
        signal, fs = ecg_signal
        config = PreprocessingConfig()
        result = ECGAnalyzer.analyze(signal, fs, config)
        assert isinstance(result["warnings"], list)


# ---------------------------------------------------------------------------
# Test 5 – flat signal (all zeros) edge case
# ---------------------------------------------------------------------------

class TestFlatSignal:
    def test_flat_signal_does_not_crash_analyze(self):
        """A flat signal should return gracefully, not raise an exception."""
        flat = np.zeros(2500)  # 5 s at 500 Hz
        config = PreprocessingConfig()
        result = ECGAnalyzer.analyze(flat, fs=500, config=config)
        assert isinstance(result, dict), "analyze() must return a dict even for flat signal"

    def test_flat_signal_has_warnings(self):
        flat = np.zeros(2500)
        config = PreprocessingConfig()
        result = ECGAnalyzer.analyze(flat, fs=500, config=config)
        # Either n_peaks == 0 or there is a warning about poor peak detection
        has_warning = len(result["warnings"]) > 0 or result["n_peaks"] == 0
        assert has_warning, "Flat signal should produce zero peaks or a warning"

    def test_detect_peaks_flat_does_not_crash(self):
        """detect_peaks on flat signal may raise ValueError but must not crash Python."""
        flat = np.zeros(2500)
        try:
            peaks = ECGAnalyzer.detect_peaks(flat, fs=500)
            # If it returns, peaks should be an array (possibly empty)
            assert isinstance(peaks, np.ndarray)
        except ValueError:
            pass  # acceptable outcome for flat signal

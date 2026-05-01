"""
Test suite for PPGAnalyzer — peak detection, vascular indices,
end-to-end analyze(), and short-signal edge cases.
"""
import sys
import os
import warnings

import numpy as np
import pytest

# sys.path already patched by conftest.py
from core.ppg_analyzer import PPGAnalyzer
from models.signal import PreprocessingConfig


# ---------------------------------------------------------------------------
# Test 1 – detect_peaks
# ---------------------------------------------------------------------------

class TestDetectPeaks:
    def test_returns_nonempty_array(self, ppg_signal):
        signal, fs = ppg_signal
        peaks = PPGAnalyzer.detect_peaks(signal, fs)
        assert isinstance(peaks, np.ndarray), "Should return np.ndarray"
        assert len(peaks) > 0, "Peaks array should not be empty"

    def test_at_least_15_peaks_in_30s(self, ppg_signal):
        """30-second PPG at 70 bpm should yield >= 15 systolic peaks."""
        signal, fs = ppg_signal
        peaks = PPGAnalyzer.detect_peaks(signal, fs)
        assert len(peaks) >= 15, (
            f"Expected >= 15 PPG peaks in 30 s, got {len(peaks)}"
        )

    def test_peaks_within_signal_bounds(self, ppg_signal):
        signal, fs = ppg_signal
        peaks = PPGAnalyzer.detect_peaks(signal, fs)
        assert np.all(peaks >= 0)
        assert np.all(peaks < len(signal))

    def test_peaks_dtype_integer(self, ppg_signal):
        signal, fs = ppg_signal
        peaks = PPGAnalyzer.detect_peaks(signal, fs)
        assert np.issubdtype(peaks.dtype, np.integer)

    def test_peaks_sorted_ascending(self, ppg_signal):
        signal, fs = ppg_signal
        peaks = PPGAnalyzer.detect_peaks(signal, fs)
        assert np.all(np.diff(peaks) > 0), "Peaks must be strictly ascending"


# ---------------------------------------------------------------------------
# Test 2 – compute_vascular_indices
# ---------------------------------------------------------------------------

class TestComputeVascularIndices:
    def test_returns_dict_with_pulse_amplitude(self, ppg_signal):
        signal, fs = ppg_signal
        peaks = PPGAnalyzer.detect_peaks(signal, fs)
        result = PPGAnalyzer.compute_vascular_indices(signal, fs, peaks)
        assert isinstance(result, dict), "Should return a dict"
        assert "pulse_amplitude" in result, "Missing key: pulse_amplitude"

    def test_pulse_amplitude_is_positive(self, ppg_signal):
        signal, fs = ppg_signal
        peaks = PPGAnalyzer.detect_peaks(signal, fs)
        result = PPGAnalyzer.compute_vascular_indices(signal, fs, peaks)
        if result["pulse_amplitude"] is not None:
            assert result["pulse_amplitude"] > 0, (
                f"Pulse amplitude should be positive, got {result['pulse_amplitude']}"
            )

    def test_returns_augmentation_index_key(self, ppg_signal):
        signal, fs = ppg_signal
        peaks = PPGAnalyzer.detect_peaks(signal, fs)
        result = PPGAnalyzer.compute_vascular_indices(signal, fs, peaks)
        assert "augmentation_index" in result

    def test_returns_respiratory_rate_key(self, ppg_signal):
        signal, fs = ppg_signal
        peaks = PPGAnalyzer.detect_peaks(signal, fs)
        result = PPGAnalyzer.compute_vascular_indices(signal, fs, peaks)
        assert "respiratory_rate" in result

    def test_empty_peaks_returns_none_values(self, ppg_signal):
        signal, fs = ppg_signal
        empty_peaks = np.array([], dtype=np.int64)
        result = PPGAnalyzer.compute_vascular_indices(signal, fs, empty_peaks)
        assert result["pulse_amplitude"] is None

    def test_respiratory_rate_in_plausible_range(self, ppg_signal):
        """Respiratory rate estimate should be between 3 and 40 breaths/min."""
        signal, fs = ppg_signal
        peaks = PPGAnalyzer.detect_peaks(signal, fs)
        result = PPGAnalyzer.compute_vascular_indices(signal, fs, peaks)
        rr = result.get("respiratory_rate")
        if rr is not None:
            assert 3.0 <= rr <= 40.0, (
                f"Respiratory rate {rr} bpm outside plausible range [3, 40]"
            )


# ---------------------------------------------------------------------------
# Test 3 – analyze() end-to-end
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_returns_dict_with_required_keys(self, ppg_signal):
        signal, fs = ppg_signal
        config = PreprocessingConfig(highcut=20.0)
        result = PPGAnalyzer.analyze(signal, fs, config)
        for key in ("n_peaks", "rr_intervals_ms", "hrv_time", "hrv_freq",
                    "vascular", "warnings"):
            assert key in result, f"Missing key in analyze() output: {key}"

    def test_n_peaks_positive(self, ppg_signal):
        signal, fs = ppg_signal
        config = PreprocessingConfig(highcut=20.0)
        result = PPGAnalyzer.analyze(signal, fs, config)
        assert result["n_peaks"] > 0

    def test_hrv_time_is_dict_or_none(self, ppg_signal):
        signal, fs = ppg_signal
        config = PreprocessingConfig(highcut=20.0)
        result = PPGAnalyzer.analyze(signal, fs, config)
        assert result["hrv_time"] is None or isinstance(result["hrv_time"], dict)

    def test_hrv_time_mean_hr_plausible(self, ppg_signal):
        """Mean heart rate should be between 40 and 150 bpm for simulated signal."""
        signal, fs = ppg_signal
        config = PreprocessingConfig(highcut=20.0)
        result = PPGAnalyzer.analyze(signal, fs, config)
        if result["hrv_time"] is not None:
            hr = result["hrv_time"].get("mean_hr")
            if hr is not None and np.isfinite(hr):
                assert 40.0 <= hr <= 150.0, f"mean_hr {hr} outside expected range"

    def test_vascular_is_dict_or_none(self, ppg_signal):
        signal, fs = ppg_signal
        config = PreprocessingConfig(highcut=20.0)
        result = PPGAnalyzer.analyze(signal, fs, config)
        assert result["vascular"] is None or isinstance(result["vascular"], dict)

    def test_vascular_pulse_amplitude_when_present(self, ppg_signal):
        signal, fs = ppg_signal
        config = PreprocessingConfig(highcut=20.0)
        result = PPGAnalyzer.analyze(signal, fs, config)
        if result["vascular"] is not None:
            assert "pulse_amplitude" in result["vascular"]

    def test_rr_intervals_are_list(self, ppg_signal):
        signal, fs = ppg_signal
        config = PreprocessingConfig(highcut=20.0)
        result = PPGAnalyzer.analyze(signal, fs, config)
        assert isinstance(result["rr_intervals_ms"], list)

    def test_warnings_is_list(self, ppg_signal):
        signal, fs = ppg_signal
        config = PreprocessingConfig(highcut=20.0)
        result = PPGAnalyzer.analyze(signal, fs, config)
        assert isinstance(result["warnings"], list)


# ---------------------------------------------------------------------------
# Test 4 – short signal (5 s) edge case
# ---------------------------------------------------------------------------

class TestShortSignal:
    def test_short_ppg_does_not_crash(self):
        """5-second PPG should complete without raising unhandled exception."""
        import neurokit2 as nk
        sig = nk.ppg_simulate(duration=5, sampling_rate=125, heart_rate=70)
        signal = np.array(sig)
        config = PreprocessingConfig(highcut=20.0)
        result = PPGAnalyzer.analyze(signal, fs=125, config=config)
        assert isinstance(result, dict)

    def test_short_ppg_produces_warning_or_low_peaks(self):
        """5-second PPG has very few beats; expect a warning or n_peaks < 10."""
        import neurokit2 as nk
        sig = nk.ppg_simulate(duration=5, sampling_rate=125, heart_rate=70)
        signal = np.array(sig)
        config = PreprocessingConfig(highcut=20.0)
        result = PPGAnalyzer.analyze(signal, fs=125, config=config)
        low_peaks = result["n_peaks"] < 10
        has_warning = len(result["warnings"]) > 0
        assert low_peaks or has_warning, (
            "Short PPG should produce a warning or < 10 peaks"
        )

    def test_very_short_ppg_graceful_hrv_skip(self):
        """If too few peaks → HRV computation should be skipped, not raise."""
        import neurokit2 as nk
        sig = nk.ppg_simulate(duration=3, sampling_rate=125, heart_rate=70)
        signal = np.array(sig)
        config = PreprocessingConfig(highcut=20.0)
        result = PPGAnalyzer.analyze(signal, fs=125, config=config)
        # hrv_time may be None when peaks are insufficient
        # Just verify the call does not crash
        assert "hrv_time" in result

    def test_fixture_short_ecg_also_works(self, short_ecg):
        """Ensure the short_ecg fixture itself works end-to-end with ECGAnalyzer."""
        from core.ecg_analyzer import ECGAnalyzer
        signal, fs = short_ecg
        config = PreprocessingConfig()
        result = ECGAnalyzer.analyze(signal, fs, config)
        assert isinstance(result, dict)

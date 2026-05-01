"""
Signal Quality Index (SQI) computation for ECG and PPG signals.
Returns a 0-100 score and detailed quality metrics.
"""
import numpy as np
from scipy import signal as scipy_signal
from typing import Optional


class SignalQualityAnalyzer:

    @staticmethod
    def compute_snr(sig: np.ndarray, fs: int, signal_type: str = "ECG") -> float:
        """Estimate SNR via band-pass power ratio."""
        nyq = fs / 2
        if signal_type.upper() == "ECG":
            low, high = 0.5 / nyq, min(40.0 / nyq, 0.99)
        else:
            low, high = 0.5 / nyq, min(8.0 / nyq, 0.99)
        try:
            sos_bp = scipy_signal.butter(4, [low, high], btype='band', output='sos')
            sig_filtered = scipy_signal.sosfilt(sos_bp, sig)
            signal_power = float(np.mean(sig_filtered ** 2))
            noise = sig - sig_filtered
            noise_power = float(np.mean(noise ** 2))
            if noise_power == 0:
                return 40.0
            snr_db = 10 * np.log10(signal_power / noise_power) if signal_power > 0 else -10.0
            return round(float(snr_db), 2)
        except Exception:
            return 0.0

    @staticmethod
    def detect_flatline(sig: np.ndarray, window_s: float = 1.0, fs: int = 500,
                        threshold: float = 1e-4) -> float:
        """Returns fraction of signal that is flatline (std < threshold)."""
        window = max(1, int(window_s * fs))
        n_flat = 0
        for i in range(0, len(sig) - window, window):
            if np.std(sig[i:i + window]) < threshold:
                n_flat += 1
        total_windows = len(sig) // window
        return float(n_flat / total_windows) if total_windows > 0 else 0.0

    @staticmethod
    def detect_clipping(sig: np.ndarray, percentile: float = 99.5) -> float:
        """Returns fraction of samples at or near the saturation level."""
        high_threshold = np.percentile(np.abs(sig), percentile)
        if high_threshold == 0:
            return 0.0
        clipped = np.sum(np.abs(sig) >= high_threshold * 0.99)
        return float(clipped / len(sig))

    @staticmethod
    def detect_baseline_wander(sig: np.ndarray, fs: int) -> float:
        """Estimate baseline wander power as fraction of total power."""
        try:
            nyq = fs / 2
            cutoff = min(0.5 / nyq, 0.49)
            sos_lp = scipy_signal.butter(4, cutoff, btype='low', output='sos')
            baseline = scipy_signal.sosfilt(sos_lp, sig)
            baseline_power = float(np.mean(baseline ** 2))
            total_power = float(np.mean(sig ** 2))
            if total_power == 0:
                return 0.0
            return round(float(baseline_power / total_power), 4)
        except Exception:
            return 0.0

    @staticmethod
    def detect_high_frequency_noise(sig: np.ndarray, fs: int, signal_type: str = "ECG") -> float:
        """Fraction of power above the expected signal band."""
        try:
            nyq = fs / 2
            cutoff = min((50.0 if signal_type.upper() == "ECG" else 10.0) / nyq, 0.99)
            sos_hp = scipy_signal.butter(4, cutoff, btype='high', output='sos')
            hf = scipy_signal.sosfilt(sos_hp, sig)
            hf_power = float(np.mean(hf ** 2))
            total_power = float(np.mean(sig ** 2))
            if total_power == 0:
                return 0.0
            return round(float(hf_power / total_power), 4)
        except Exception:
            return 0.0

    @classmethod
    def compute_sqi(cls, sig: np.ndarray, fs: int, signal_type: str = "ECG") -> dict:
        """
        Compute overall Signal Quality Index (0–100) and component metrics.

        Score interpretation:
            ≥ 80 = Good
            60–79 = Acceptable
            40–59 = Poor
            < 40  = Unacceptable
        """
        snr = cls.compute_snr(sig, fs, signal_type)
        flatline_frac = cls.detect_flatline(sig, fs=fs)
        clipping_frac = cls.detect_clipping(sig)
        baseline_frac = cls.detect_baseline_wander(sig, fs)
        hf_noise_frac = cls.detect_high_frequency_noise(sig, fs, signal_type)

        # Score components (each 0–100, then weighted average)
        snr_score = min(100, max(0, (snr + 5) * 4))          # -5 dB=0, 20 dB=100
        flatline_score = max(0, 100 - flatline_frac * 200)
        clipping_score = max(0, 100 - clipping_frac * 300)
        baseline_score = max(0, 100 - baseline_frac * 150)
        noise_score = max(0, 100 - hf_noise_frac * 200)

        overall = (
            0.35 * snr_score
            + 0.25 * flatline_score
            + 0.15 * clipping_score
            + 0.15 * baseline_score
            + 0.10 * noise_score
        )
        overall = round(float(overall), 1)

        if overall >= 80:
            quality_label = "Good"
        elif overall >= 60:
            quality_label = "Acceptable"
        elif overall >= 40:
            quality_label = "Poor"
        else:
            quality_label = "Unacceptable"

        warnings: list[str] = []
        if flatline_frac > 0.1:
            warnings.append(f"Flatline detected in {flatline_frac * 100:.0f}% of signal")
        if clipping_frac > 0.05:
            warnings.append(f"Signal clipping in {clipping_frac * 100:.1f}% of samples")
        if baseline_frac > 0.3:
            warnings.append("Significant baseline wander detected")
        if hf_noise_frac > 0.4:
            warnings.append("High-frequency noise exceeds 40% of signal power")
        if snr < 5:
            warnings.append(f"Low SNR: {snr:.1f} dB")

        return {
            "overall_score": overall,
            "quality_label": quality_label,
            "snr_db": snr,
            "flatline_fraction": round(flatline_frac, 4),
            "clipping_fraction": round(clipping_frac, 4),
            "baseline_wander_fraction": round(baseline_frac, 4),
            "hf_noise_fraction": round(hf_noise_frac, 4),
            "warnings": warnings,
        }

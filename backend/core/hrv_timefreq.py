"""
Time-Frequency Analysis for HRV — STFT and CWT (Morlet wavelet).
Equivalent to spettrogramma HRV tempo-frequenza.
Uses scipy.signal only — no additional dependencies.
"""
import numpy as np
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d
from typing import Optional


class TimeFrequencyAnalyzer:

    @staticmethod
    def _resample_rr(rr: np.ndarray, fs_resample: float = 4.0) -> tuple:
        """Resample RR series to uniform grid via cubic interpolation."""
        if len(rr) < 4:
            return None, None
        t_rr = np.cumsum(rr) / 1000.0
        t_rr = t_rr - t_rr[0]
        total_s = t_rr[-1]
        if total_s < 10.0:
            return None, None
        t_uniform = np.linspace(0, total_s, int(total_s * fs_resample))
        interp = interp1d(t_rr, rr, kind="cubic", bounds_error=False, fill_value="extrapolate")
        return t_uniform, interp(t_uniform)

    @classmethod
    def compute_stft(cls, rr: np.ndarray, fs_resample: float = 4.0,
                     window_s: float = 60.0, overlap: float = 0.75) -> Optional[dict]:
        """
        Short-Time Fourier Transform of RR tachogram.
        Returns time-frequency power matrix for LF/HF band tracking.
        """
        t_uniform, rr_uniform = cls._resample_rr(rr, fs_resample)
        if t_uniform is None:
            return None
        try:
            nperseg = int(window_s * fs_resample)
            nperseg = min(nperseg, len(rr_uniform) // 2)
            nperseg = max(nperseg, 16)
            noverlap = int(nperseg * overlap)
            noverlap = min(noverlap, nperseg - 1)

            freqs, times, Zxx = scipy_signal.stft(
                rr_uniform, fs=fs_resample,
                window="hann", nperseg=nperseg, noverlap=noverlap,
            )
            power = np.abs(Zxx) ** 2  # shape: (freq, time)

            # Extract band powers over time
            lf_mask = (freqs >= 0.04) & (freqs < 0.15)
            hf_mask = (freqs >= 0.15) & (freqs < 0.40)
            lf_over_time = np.sum(power[lf_mask, :], axis=0).tolist()
            hf_over_time = np.sum(power[hf_mask, :], axis=0).tolist()
            lf_hf_over_time = [
                float(lf / hf) if hf > 0 else None
                for lf, hf in zip(lf_over_time, hf_over_time)
            ]

            # Clip to HRV-relevant frequencies (0-0.5 Hz)
            freq_mask = freqs <= 0.50
            return {
                "times": (times + t_uniform[0]).tolist(),
                "freqs": freqs[freq_mask].tolist(),
                "power_db": (10 * np.log10(power[freq_mask, :] + 1e-12)).tolist(),
                "lf_over_time": lf_over_time,
                "hf_over_time": hf_over_time,
                "lf_hf_over_time": lf_hf_over_time,
            }
        except Exception:
            return None

    @classmethod
    def compute_cwt(cls, rr: np.ndarray, fs_resample: float = 4.0) -> Optional[dict]:
        """
        Continuous Wavelet Transform (Morlet) scalogram.
        Provides adaptive time-frequency resolution (better than STFT for non-stationary HRV).
        """
        t_uniform, rr_uniform = cls._resample_rr(rr, fs_resample)
        if t_uniform is None:
            return None
        try:
            # Frequencies of interest: 0.01 Hz to 0.5 Hz
            freqs_target = np.logspace(np.log10(0.01), np.log10(0.50), 40)
            # Convert to wavelet widths: w = f * fs / f0 where f0=1 for Morlet
            # scipy.signal.cwt uses widths directly for morlet wavelet
            widths = fs_resample / freqs_target  # approximate scale-frequency relation

            cwt_matrix = scipy_signal.cwt(rr_uniform, scipy_signal.morlet2, widths,
                                           dtype=complex if hasattr(scipy_signal, 'morlet2') else float)
            power = np.abs(cwt_matrix) ** 2  # shape: (scale, time)

            # Band powers over time
            lf_idx = (freqs_target >= 0.04) & (freqs_target < 0.15)
            hf_idx = (freqs_target >= 0.15) & (freqs_target < 0.40)
            lf_over_time = np.mean(power[lf_idx, :], axis=0).tolist() if lf_idx.any() else []
            hf_over_time = np.mean(power[hf_idx, :], axis=0).tolist() if hf_idx.any() else []

            # Downsample matrix for frontend (max 200 time points)
            n_t = power.shape[1]
            step = max(1, n_t // 200)
            power_ds = power[:, ::step]
            t_ds = t_uniform[::step]

            return {
                "times": t_ds.tolist(),
                "freqs": freqs_target.tolist(),
                "power_db": (10 * np.log10(power_ds + 1e-12)).tolist(),
                "lf_over_time": lf_over_time[::step] if lf_over_time else [],
                "hf_over_time": hf_over_time[::step] if hf_over_time else [],
            }
        except Exception:
            return None

    @classmethod
    def analyze(cls, rr: np.ndarray) -> dict:
        """Compute both STFT and CWT. Returns dict with both results."""
        return {
            "stft": cls.compute_stft(rr),
            "cwt": cls.compute_cwt(rr),
            "has_data": len(rr) >= 20,
        }

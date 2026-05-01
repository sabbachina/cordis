"""
Signal Preprocessor — filtering and normalization pipeline for ECG/PPG signals.

All public methods accept and return np.ndarray (1-D float64).
"""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from models.signal import PreprocessingConfig


class SignalPreprocessor:
    """
    Stateless collection of preprocessing steps.

    Typical pipeline
    ----------------
    1. remove_baseline_wander   (highpass > 0.5 Hz)
    2. bandpass_filter          (e.g. 0.5 – 40 Hz for ECG)
    3. remove_powerline         (notch 50 Hz + 60 Hz)
    4. normalize                (z-score)
    """

    # ------------------------------------------------------------------
    # Band-pass (Butterworth)
    # ------------------------------------------------------------------
    @staticmethod
    def bandpass_filter(
        signal: np.ndarray,
        lowcut: float,
        highcut: float,
        fs: int,
        order: int = 4,
    ) -> np.ndarray:
        """
        Apply a zero-phase Butterworth band-pass filter.

        Parameters
        ----------
        signal  : 1-D signal array
        lowcut  : lower cut-off frequency in Hz
        highcut : upper cut-off frequency in Hz
        fs      : sampling rate in Hz
        order   : filter order (default 4)
        """
        nyq = fs / 2.0
        if lowcut <= 0:
            raise ValueError(f"lowcut must be > 0; got {lowcut}.")
        if highcut >= nyq:
            raise ValueError(
                f"highcut ({highcut} Hz) must be strictly less than Nyquist "
                f"({nyq} Hz) for fs={fs} Hz."
            )
        if lowcut >= highcut:
            raise ValueError(
                f"lowcut ({lowcut} Hz) must be less than highcut ({highcut} Hz)."
            )

        low = lowcut / nyq
        high = highcut / nyq

        sos = scipy_signal.butter(order, [low, high], btype="bandpass", output="sos")
        filtered = scipy_signal.sosfiltfilt(sos, signal)
        return filtered.astype(np.float64)

    # ------------------------------------------------------------------
    # Baseline wander removal (highpass)
    # ------------------------------------------------------------------
    @staticmethod
    def remove_baseline_wander(
        signal: np.ndarray,
        fs: int,
        cutoff: float = 0.5,
        order: int = 4,
    ) -> np.ndarray:
        """
        Remove low-frequency baseline wander with a high-pass Butterworth filter.

        Parameters
        ----------
        signal : 1-D signal array
        fs     : sampling rate in Hz
        cutoff : high-pass cut-off frequency (default 0.5 Hz)
        order  : filter order (default 4)
        """
        nyq = fs / 2.0
        if cutoff <= 0 or cutoff >= nyq:
            raise ValueError(
                f"cutoff ({cutoff} Hz) must be in (0, {nyq}) Hz for fs={fs}."
            )

        norm_cutoff = cutoff / nyq
        sos = scipy_signal.butter(order, norm_cutoff, btype="highpass", output="sos")
        filtered = scipy_signal.sosfiltfilt(sos, signal)
        return filtered.astype(np.float64)

    # ------------------------------------------------------------------
    # Powerline interference removal (notch)
    # ------------------------------------------------------------------
    @staticmethod
    def remove_powerline(
        signal: np.ndarray,
        fs: int,
        freqs: tuple[float, ...] = (50.0, 60.0),
        quality_factor: float = 30.0,
    ) -> np.ndarray:
        """
        Remove powerline interference with one or more IIR notch filters.

        Parameters
        ----------
        signal         : 1-D signal array
        fs             : sampling rate in Hz
        freqs          : notch frequencies to suppress (default: 50 Hz and 60 Hz)
        quality_factor : Q-factor controlling notch bandwidth (higher → narrower)
        """
        nyq = fs / 2.0
        result = signal.astype(np.float64).copy()

        for freq in freqs:
            if freq <= 0 or freq >= nyq:
                # Skip frequencies that are out of range for this fs
                continue
            b, a = scipy_signal.iirnotch(freq, quality_factor, fs=float(fs))
            result = scipy_signal.filtfilt(b, a, result)

        return result

    # ------------------------------------------------------------------
    # Normalization (z-score)
    # ------------------------------------------------------------------
    @staticmethod
    def normalize(signal: np.ndarray) -> np.ndarray:
        """
        Apply zero-mean, unit-variance (z-score) normalization.

        If the signal has zero standard deviation (constant signal), it is
        returned as-is to avoid division by zero.
        """
        std = np.std(signal)
        if std == 0.0:
            return signal.astype(np.float64)
        return ((signal - np.mean(signal)) / std).astype(np.float64)

    # ------------------------------------------------------------------
    # Savitzky-Golay smoothing
    # ------------------------------------------------------------------
    @staticmethod
    def savgol_filter(
        signal: np.ndarray,
        window_length_sec: float = 0.1,
        fs: int = 250,
        polyorder: int = 3,
    ) -> np.ndarray:
        """
        Apply Savitzky-Golay smoothing as an alternative to Butterworth.

        Parameters
        ----------
        signal             : 1-D signal array
        window_length_sec  : window length in seconds (converted to samples)
        fs                 : sampling rate in Hz
        polyorder          : polynomial order for the filter
        """
        window_samples = int(window_length_sec * fs)
        # Must be odd and > polyorder
        if window_samples % 2 == 0:
            window_samples += 1
        window_samples = max(window_samples, polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3)

        filtered = scipy_signal.savgol_filter(signal, window_samples, polyorder)
        return filtered.astype(np.float64)

    # ------------------------------------------------------------------
    # Main pipeline entry point
    # ------------------------------------------------------------------
    @staticmethod
    def preprocess(
        raw_signal: np.ndarray,
        fs: int,
        config: PreprocessingConfig,
    ) -> np.ndarray:
        """
        Apply the full preprocessing pipeline according to *config*.

        Pipeline order
        --------------
        1. Baseline wander removal       (if config.remove_baseline is True)
        2. Band-pass / smoothing filter  (method-dependent)
        3. Powerline notch filter        (always applied)
        4. Z-score normalization         (always applied as final step)

        Parameters
        ----------
        raw_signal : 1-D NumPy array of raw amplitude values
        fs         : sampling rate in Hz
        config     : PreprocessingConfig instance

        Returns
        -------
        Preprocessed 1-D NumPy array (float64)
        """
        sig = raw_signal.astype(np.float64).copy()

        # ---- step 1: baseline wander removal ---------------------------
        if config.remove_baseline:
            sig = SignalPreprocessor.remove_baseline_wander(sig, fs, cutoff=0.5)

        # ---- step 2: band-pass / smoothing filter ----------------------
        method = config.method.lower()
        nyq = fs / 2.0
        highcut_safe = min(config.highcut, nyq * 0.99)  # guard against Nyquist

        if method == "butterworth":
            sig = SignalPreprocessor.bandpass_filter(
                sig,
                lowcut=config.lowcut,
                highcut=highcut_safe,
                fs=fs,
            )
        elif method == "savgol":
            # Savitzky-Golay: apply high-pass first (via remove_baseline_wander
            # with user-specified lowcut), then smooth the HF part
            sig = SignalPreprocessor.remove_baseline_wander(
                sig, fs, cutoff=config.lowcut
            )
            sig = SignalPreprocessor.savgol_filter(sig, fs=fs)
        elif method == "neurokit":
            # Use the same Butterworth approach but with neurokit2-recommended
            # defaults; fall back gracefully if neurokit2 is unavailable
            try:
                import neurokit2 as nk  # type: ignore

                sig = nk.signal_filter(
                    sig,
                    sampling_rate=fs,
                    lowcut=config.lowcut,
                    highcut=highcut_safe,
                    method="butterworth",
                    order=4,
                )
                sig = sig.astype(np.float64)
            except Exception:
                # Graceful fallback to SciPy Butterworth
                sig = SignalPreprocessor.bandpass_filter(
                    sig,
                    lowcut=config.lowcut,
                    highcut=highcut_safe,
                    fs=fs,
                )
        else:
            raise ValueError(
                f"Unknown preprocessing method '{method}'. "
                "Choose one of: butterworth, savgol, neurokit."
            )

        # ---- step 3: powerline removal ---------------------------------
        sig = SignalPreprocessor.remove_powerline(sig, fs)

        # ---- step 4: normalization -------------------------------------
        sig = SignalPreprocessor.normalize(sig)

        return sig

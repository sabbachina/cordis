"""
PPG Analyzer — peak detection, HRV and vascular indices for PPG signals.

Reuses the shared HRV computation logic from ecg_analyzer internals.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d

from models.signal import PreprocessingConfig

# Re-use the shared HRV helper functions defined in ecg_analyzer
from core.ecg_analyzer import _compute_hrv_time_domain, _compute_hrv_freq_domain


class PPGAnalyzer:
    """
    Feature extractor for PPG (photoplethysmography) signals.
    """

    # ------------------------------------------------------------------
    # Systolic peak detection
    # ------------------------------------------------------------------
    @staticmethod
    def detect_peaks(
        signal: np.ndarray,
        fs: int,
    ) -> np.ndarray:
        """
        Detect systolic peaks in a preprocessed PPG signal using NeuroKit2.

        Parameters
        ----------
        signal : 1-D preprocessed PPG array
        fs     : sampling rate in Hz

        Returns
        -------
        1-D integer array of peak sample indices
        """
        try:
            import neurokit2 as nk  # type: ignore

            ppg_signals, ppg_info = nk.ppg_process(signal, sampling_rate=fs)
            peaks = ppg_info["PPG_Peaks"]
            return np.asarray(peaks, dtype=np.int64)

        except Exception as exc:
            raise ValueError(f"PPG peak detection failed: {exc}") from exc

    # ------------------------------------------------------------------
    # HRV — time domain (delegates to shared helper)
    # ------------------------------------------------------------------
    @staticmethod
    def compute_hrv_time(rr_intervals_ms: np.ndarray) -> dict:
        """
        Compute time-domain HRV metrics from inter-beat intervals.

        Parameters
        ----------
        rr_intervals_ms : 1-D array of peak-to-peak intervals in ms

        Returns
        -------
        dict  →  mean_hr (bpm), sdnn (ms), rmssd (ms), pnn50 (%), pnn20 (%)
        """
        return _compute_hrv_time_domain(rr_intervals_ms)

    # ------------------------------------------------------------------
    # Vascular indices
    # ------------------------------------------------------------------
    @staticmethod
    def compute_vascular_indices(
        signal: np.ndarray,
        fs: int,
        peaks: np.ndarray,
    ) -> dict:
        """
        Compute PPG-specific vascular biomarkers.

        Metrics
        -------
        pulse_amplitude    : mean peak-to-trough amplitude (a.u.)
        augmentation_index : ratio of second systolic peak to first (AIx), %
        respiratory_rate   : estimated from envelope modulation (breaths/min)

        Parameters
        ----------
        signal : preprocessed PPG signal (z-score normalised)
        fs     : sampling rate in Hz
        peaks  : systolic peak indices

        Returns
        -------
        dict with keys: pulse_amplitude, augmentation_index, respiratory_rate
        """
        result: dict = {
            "pulse_amplitude": None,
            "augmentation_index": None,
            "respiratory_rate": None,
        }

        if peaks is None or len(peaks) < 2:
            return result

        # ----------------------------------------------------------------
        # 1. Pulse amplitude — mean(peak_value - preceding_trough_value)
        # ----------------------------------------------------------------
        amplitudes: list[float] = []
        for i, pk in enumerate(peaks):
            # Trough = minimum in the window before this peak
            start = int(peaks[i - 1]) if i > 0 else 0
            end = int(pk)
            if end <= start:
                continue
            trough_val = float(np.min(signal[start:end]))
            peak_val = float(signal[int(pk)])
            amplitudes.append(peak_val - trough_val)

        if amplitudes:
            result["pulse_amplitude"] = float(np.mean(amplitudes))

        # ----------------------------------------------------------------
        # 2. Augmentation Index (AIx)
        #    For each beat, find P1 (first systolic peak) and P2 (inflection
        #    point or second peak).  AIx = (P2 - P1) / pulse_pressure * 100
        #    Here we use a simplified approach: search for a secondary maximum
        #    in the first third of the diastolic period after the systolic peak.
        # ----------------------------------------------------------------
        aix_values: list[float] = []

        for i in range(len(peaks) - 1):
            pk_idx = int(peaks[i])
            next_pk_idx = int(peaks[i + 1])
            beat_len = next_pk_idx - pk_idx
            if beat_len < 6:
                continue

            beat = signal[pk_idx: next_pk_idx]
            # Window for second peak: 20 % – 60 % of beat length after P1
            search_start = max(1, int(0.20 * beat_len))
            search_end = min(beat_len - 1, int(0.60 * beat_len))

            if search_end <= search_start:
                continue

            # P1 is the systolic peak value
            p1 = float(signal[pk_idx])

            # P2 candidate: local maximum in search window
            search_window = beat[search_start: search_end]
            local_max_idx = int(np.argmax(search_window)) + search_start
            p2 = float(beat[local_max_idx])

            # Trough before systolic peak (diastolic baseline)
            trough_win_start = int(peaks[i - 1]) if i > 0 else 0
            baseline = float(np.min(signal[trough_win_start: pk_idx + 1]))

            pulse_pressure = p1 - baseline
            if pulse_pressure <= 0:
                continue

            aix = (p2 - p1) / pulse_pressure * 100.0
            aix_values.append(aix)

        if aix_values:
            result["augmentation_index"] = float(np.mean(aix_values))

        # ----------------------------------------------------------------
        # 3. Respiratory rate via amplitude modulation of the PPG envelope
        #    Method: extract peak amplitudes, resample to uniform grid,
        #    FFT → dominant frequency in 0.1–0.5 Hz (6–30 breaths/min)
        # ----------------------------------------------------------------
        if len(peaks) >= 6:
            try:
                peak_times = peaks.astype(np.float64) / fs          # seconds
                peak_amplitudes = signal[peaks].astype(np.float64)

                # Interpolate envelope onto a 4 Hz grid
                t_uniform = np.arange(peak_times[0], peak_times[-1], 1.0 / 4.0)
                if t_uniform.size >= 8:
                    interpolator = interp1d(
                        peak_times,
                        peak_amplitudes,
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    env_uniform = interpolator(t_uniform)

                    # Detrend and window before FFT
                    env_detrended = env_uniform - np.mean(env_uniform)
                    window = np.hanning(len(env_detrended))
                    env_windowed = env_detrended * window

                    fft_vals = np.abs(np.fft.rfft(env_windowed))
                    fft_freqs = np.fft.rfftfreq(len(env_windowed), d=1.0 / 4.0)

                    # Search for dominant frequency between 0.1 and 0.5 Hz
                    resp_mask = (fft_freqs >= 0.1) & (fft_freqs <= 0.5)
                    if resp_mask.any():
                        dominant_freq = fft_freqs[resp_mask][
                            np.argmax(fft_vals[resp_mask])
                        ]
                        rr_bpm = dominant_freq * 60.0
                        result["respiratory_rate"] = float(rr_bpm)
            except Exception:
                pass  # respiratory rate is best-effort

        return result

    # ------------------------------------------------------------------
    # Main orchestrator
    # ------------------------------------------------------------------
    @staticmethod
    def analyze(
        signal: np.ndarray,
        fs: int,
        config: PreprocessingConfig,
        compute_hrv: bool = True,
        compute_vascular: bool = True,
    ) -> dict:
        """
        Full PPG analysis pipeline.

        Parameters
        ----------
        signal           : preprocessed 1-D PPG array
        fs               : sampling rate in Hz
        config           : PreprocessingConfig (context only here)
        compute_hrv      : whether to compute HRV metrics
        compute_vascular : whether to compute vascular indices

        Returns
        -------
        dict with keys:
            n_peaks, rr_intervals_ms,
            hrv_time (dict), hrv_freq (dict),
            vascular (dict),
            warnings (list[str])
        """
        result: dict = {
            "n_peaks": 0,
            "rr_intervals_ms": [],
            "hrv_time": None,
            "hrv_freq": None,
            "vascular": None,
            "warnings": [],
        }

        # ---- Peak detection -------------------------------------------
        try:
            peaks = PPGAnalyzer.detect_peaks(signal, fs)
        except ValueError as exc:
            result["warnings"].append(f"Peak detection error: {exc}")
            return result

        result["n_peaks"] = int(len(peaks))

        # Guard: fewer than 5 peaks → degrade gracefully
        if len(peaks) < 5:
            result["warnings"].append(
                f"Only {len(peaks)} PPG peak(s) detected (minimum 5 required). "
                "Signal may be too short, noisy, or incorrectly formatted. "
                "HRV and vascular indices will not be computed."
            )
            return result

        # IBI (inter-beat interval) in ms
        ibi_ms = np.diff(peaks.astype(np.float64)) / fs * 1000.0

        # Artefact rejection: physiologically plausible HR 20–300 bpm
        valid_mask = (ibi_ms >= 200.0) & (ibi_ms <= 3000.0)
        n_rejected = int((~valid_mask).sum())
        if n_rejected > 0:
            result["warnings"].append(
                f"{n_rejected} IBI(s) outside 200–3000 ms were discarded as artefacts."
            )
        ibi_ms = ibi_ms[valid_mask]
        result["rr_intervals_ms"] = ibi_ms.tolist()

        if ibi_ms.size < 2:
            result["warnings"].append(
                "Fewer than 2 valid IBIs after artefact rejection — "
                "HRV computation skipped."
            )
            return result

        # ---- HRV time domain ------------------------------------------
        if compute_hrv:
            try:
                result["hrv_time"] = PPGAnalyzer.compute_hrv_time(ibi_ms)
            except Exception as exc:
                result["warnings"].append(f"HRV time-domain failed: {exc}")

            try:
                result["hrv_freq"] = _compute_hrv_freq_domain(ibi_ms)
            except Exception as exc:
                result["warnings"].append(f"HRV frequency-domain failed: {exc}")

        # ---- Vascular indices -----------------------------------------
        if compute_vascular:
            try:
                result["vascular"] = PPGAnalyzer.compute_vascular_indices(
                    signal, fs, peaks
                )
            except Exception as exc:
                result["warnings"].append(f"Vascular index computation failed: {exc}")
                result["vascular"] = {
                    "pulse_amplitude": None,
                    "augmentation_index": None,
                    "respiratory_rate": None,
                }

        return result

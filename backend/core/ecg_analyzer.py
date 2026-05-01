"""
ECG Analyzer — peak detection, HRV (time + frequency domain) and
morphology feature extraction for ECG signals.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d

from models.signal import PreprocessingConfig


# ---------------------------------------------------------------------------
# Internal helper — shared HRV time-domain computation
# ---------------------------------------------------------------------------

def _compute_hrv_time_domain(rr_intervals_ms: np.ndarray) -> dict:
    """
    Compute standard HRV time-domain biomarkers from an array of RR intervals
    expressed in milliseconds.

    Parameters
    ----------
    rr_intervals_ms : 1-D array of RR (or NN) intervals in ms

    Returns
    -------
    dict with keys: mean_hr, sdnn, rmssd, pnn50, pnn20
    """
    rr = np.asarray(rr_intervals_ms, dtype=np.float64)

    if rr.size < 2:
        raise ValueError(
            f"At least 2 RR intervals required; got {rr.size}."
        )

    # Mean heart rate (bpm)
    mean_rr_ms = np.mean(rr)
    mean_hr = 60_000.0 / mean_rr_ms if mean_rr_ms > 0 else np.nan

    # SDNN — standard deviation of all NN intervals
    sdnn = float(np.std(rr, ddof=1)) if rr.size > 1 else np.nan

    # Successive differences
    diff_rr = np.diff(rr)

    # RMSSD — root mean square of successive differences
    rmssd = float(np.sqrt(np.mean(diff_rr ** 2)))

    # pNN50 — percentage of successive differences > 50 ms
    pnn50 = float(np.sum(np.abs(diff_rr) > 50.0) / diff_rr.size * 100.0)

    # pNN20 — percentage of successive differences > 20 ms
    pnn20 = float(np.sum(np.abs(diff_rr) > 20.0) / diff_rr.size * 100.0)

    return {
        "mean_hr": mean_hr,
        "sdnn": sdnn,
        "rmssd": rmssd,
        "pnn50": pnn50,
        "pnn20": pnn20,
    }


# ---------------------------------------------------------------------------
# Internal helper — HRV frequency-domain computation (shared)
# ---------------------------------------------------------------------------

def _compute_hrv_freq_domain(
    rr_intervals_ms: np.ndarray,
    fs_resample: float = 4.0,
) -> dict:
    """
    Compute HRV frequency-domain biomarkers using Welch's periodogram on a
    resampled, evenly-spaced RR tachogram.

    Frequency bands
    ---------------
    VLF  : 0.003 – 0.04  Hz
    LF   : 0.04  – 0.15  Hz
    HF   : 0.15  – 0.40  Hz

    Parameters
    ----------
    rr_intervals_ms : 1-D array of RR intervals in ms
    fs_resample     : target resampling frequency for the tachogram (Hz)

    Returns
    -------
    dict with keys: vlf_power, lf_power, hf_power, lf_hf_ratio
                    (all in ms²)
    """
    rr = np.asarray(rr_intervals_ms, dtype=np.float64)

    if rr.size < 4:
        raise ValueError(
            f"Frequency-domain HRV requires at least 4 RR intervals; got {rr.size}."
        )

    # Build cumulative time axis (mid-point of each interval)
    t_rr = np.cumsum(rr) / 1000.0      # seconds
    t_rr = t_rr - t_rr[0]              # start from 0

    total_duration = t_rr[-1]
    n_resamp = int(total_duration * fs_resample)

    if n_resamp < 8:
        raise ValueError(
            "RR series too short for frequency-domain analysis "
            f"(duration {total_duration:.1f} s, need ≥ 2 s at {fs_resample} Hz)."
        )

    # Cubic interpolation to evenly-spaced tachogram
    t_uniform = np.linspace(t_rr[0], t_rr[-1], n_resamp)
    kind = "cubic" if rr.size >= 4 else "linear"
    interpolator = interp1d(t_rr, rr, kind=kind, bounds_error=False, fill_value="extrapolate")
    rr_uniform = interpolator(t_uniform)

    # Welch PSD
    nperseg = min(n_resamp, max(64, n_resamp // 4))
    freqs, psd = scipy_signal.welch(
        rr_uniform,
        fs=fs_resample,
        nperseg=nperseg,
        scaling="density",
    )

    freq_res = freqs[1] - freqs[0]  # Hz per bin

    def _band_power(f_low: float, f_high: float) -> float:
        mask = (freqs >= f_low) & (freqs < f_high)
        return float(np.trapz(psd[mask], freqs[mask])) if mask.any() else 0.0

    vlf_power = _band_power(0.003, 0.04)
    lf_power = _band_power(0.04, 0.15)
    hf_power = _band_power(0.15, 0.40)
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.nan

    return {
        "vlf_power": vlf_power,
        "lf_power": lf_power,
        "hf_power": hf_power,
        "lf_hf_ratio": lf_hf_ratio,
    }


# ---------------------------------------------------------------------------
# ECGAnalyzer
# ---------------------------------------------------------------------------

class ECGAnalyzer:
    """
    Feature extractor for ECG signals.

    Methods are designed to be called individually or through the
    high-level `analyze()` orchestrator.
    """

    # ------------------------------------------------------------------
    # R-peak detection
    # ------------------------------------------------------------------
    @staticmethod
    def detect_peaks(
        signal: np.ndarray,
        fs: int,
    ) -> np.ndarray:
        """
        Detect R-peaks in a preprocessed ECG signal using NeuroKit2.

        Parameters
        ----------
        signal : 1-D preprocessed ECG array
        fs     : sampling rate in Hz

        Returns
        -------
        1-D integer array of R-peak sample indices
        """
        try:
            import neurokit2 as nk  # type: ignore

            _, rpeaks_info = nk.ecg_peaks(signal, sampling_rate=fs, method="neurokit")
            r_peaks = rpeaks_info["ECG_R_Peaks"]
            return np.asarray(r_peaks, dtype=np.int64)

        except Exception as exc:
            raise ValueError(f"R-peak detection failed: {exc}") from exc

    # ------------------------------------------------------------------
    # HRV — time domain
    # ------------------------------------------------------------------
    @staticmethod
    def compute_hrv_time(rr_intervals_ms: np.ndarray) -> dict:
        """
        Compute time-domain HRV metrics.

        Parameters
        ----------
        rr_intervals_ms : 1-D array of RR intervals in milliseconds

        Returns
        -------
        dict  →  mean_hr (bpm), sdnn (ms), rmssd (ms), pnn50 (%), pnn20 (%)
        """
        return _compute_hrv_time_domain(rr_intervals_ms)

    # ------------------------------------------------------------------
    # HRV — frequency domain
    # ------------------------------------------------------------------
    @staticmethod
    def compute_hrv_freq(
        rr_intervals_ms: np.ndarray,
        fs_resample: float = 4.0,
    ) -> dict:
        """
        Compute frequency-domain HRV metrics (VLF, LF, HF, LF/HF).

        Parameters
        ----------
        rr_intervals_ms : 1-D array of RR intervals in ms
        fs_resample     : resampling frequency for the tachogram (Hz)

        Returns
        -------
        dict  →  vlf_power, lf_power, hf_power, lf_hf_ratio  (all in ms²)
        """
        return _compute_hrv_freq_domain(rr_intervals_ms, fs_resample=fs_resample)

    # ------------------------------------------------------------------
    # ECG morphology
    # ------------------------------------------------------------------
    @staticmethod
    def extract_morphology(
        signal: np.ndarray,
        fs: int,
        r_peaks: np.ndarray,
    ) -> dict:
        """
        Extract wave-interval ECG morphology features via NeuroKit2 delineation.

        Features
        --------
        pr_interval_ms  : PR interval (P-wave onset to QRS onset) in ms
        qrs_duration_ms : QRS complex duration in ms
        qtc_ms          : Corrected QT interval (Bazett formula) in ms
        st_deviation_mv : Mean ST-segment deviation from isoelectric baseline in mV

        Parameters
        ----------
        signal  : preprocessed ECG signal (z-score normalised)
        fs      : sampling rate in Hz
        r_peaks : array of R-peak indices

        Returns
        -------
        dict with keys: pr_interval_ms, qrs_duration_ms, qtc_ms, st_deviation_mv
                        (values may be None on delineation failure)
        """
        result: dict = {
            "pr_interval_ms": None,
            "qrs_duration_ms": None,
            "qtc_ms": None,
            "st_deviation_mv": None,
        }

        if r_peaks is None or len(r_peaks) < 3:
            return result

        try:
            import neurokit2 as nk  # type: ignore
            import pandas as pd

            _, waves_info = nk.ecg_delineate(
                signal,
                rpeaks=r_peaks,
                sampling_rate=fs,
                method="dwt",
            )

            def _safe_median_ms(indices_array) -> Optional[float]:
                """Convert sample-index differences to ms, return median."""
                if indices_array is None:
                    return None
                arr = np.asarray(indices_array, dtype=np.float64)
                arr = arr[~np.isnan(arr)]
                if arr.size == 0:
                    return None
                return float(np.median(arr) / fs * 1000.0)

            # ---- PR interval ----------------------------------------
            p_onsets = waves_info.get("ECG_P_Onsets")
            q_peaks = waves_info.get("ECG_Q_Peaks")

            if p_onsets is not None and q_peaks is not None:
                p_arr = np.asarray(p_onsets, dtype=np.float64)
                q_arr = np.asarray(q_peaks, dtype=np.float64)
                valid = ~np.isnan(p_arr) & ~np.isnan(q_arr)
                if valid.sum() > 0:
                    pr_samples = q_arr[valid] - p_arr[valid]
                    result["pr_interval_ms"] = float(
                        np.median(pr_samples) / fs * 1000.0
                    )

            # ---- QRS duration ----------------------------------------
            qrs_onsets = waves_info.get("ECG_Q_Peaks")   # proxy onset
            qrs_offsets = waves_info.get("ECG_S_Peaks")  # proxy offset

            if qrs_onsets is not None and qrs_offsets is not None:
                on_arr = np.asarray(qrs_onsets, dtype=np.float64)
                off_arr = np.asarray(qrs_offsets, dtype=np.float64)
                valid = ~np.isnan(on_arr) & ~np.isnan(off_arr)
                if valid.sum() > 0:
                    qrs_samples = np.abs(off_arr[valid] - on_arr[valid])
                    result["qrs_duration_ms"] = float(
                        np.median(qrs_samples) / fs * 1000.0
                    )

            # ---- QTc (Bazett) ----------------------------------------
            t_offsets = waves_info.get("ECG_T_Offsets")
            q_onsets_raw = waves_info.get("ECG_Q_Peaks")

            if t_offsets is not None and q_onsets_raw is not None:
                t_arr = np.asarray(t_offsets, dtype=np.float64)
                q_arr2 = np.asarray(q_onsets_raw, dtype=np.float64)
                valid = ~np.isnan(t_arr) & ~np.isnan(q_arr2)
                if valid.sum() > 0:
                    qt_samples = t_arr[valid] - q_arr2[valid]
                    qt_ms = np.median(qt_samples) / fs * 1000.0
                    # RR in seconds for Bazett: QTc = QT / sqrt(RR)
                    rr_ms = np.diff(r_peaks.astype(np.float64)) / fs * 1000.0
                    if rr_ms.size > 0:
                        rr_s = np.median(rr_ms) / 1000.0
                        qtc = qt_ms / np.sqrt(rr_s) if rr_s > 0 else np.nan
                        result["qtc_ms"] = float(qtc)

            # ---- ST deviation (mV proxy) --------------------------------
            # Measure signal amplitude at J-point (80 ms after R-peak) minus
            # baseline (signal at P-onset or 200 ms before R-peak)
            st_deviations: list[float] = []
            j_offset = int(0.08 * fs)  # 80 ms after R

            for r_idx in r_peaks:
                j_idx = int(r_idx) + j_offset
                if j_idx >= len(signal):
                    continue
                # Isoelectric reference: 200 ms before R-peak (or start)
                baseline_idx = max(0, int(r_idx) - int(0.2 * fs))
                st_val = float(signal[j_idx]) - float(signal[baseline_idx])
                st_deviations.append(st_val)

            if st_deviations:
                result["st_deviation_mv"] = float(np.mean(st_deviations))

        except Exception as exc:
            warnings.warn(f"ECG morphology delineation failed: {exc}")

        return result

    # ------------------------------------------------------------------
    # HRV non-linear metrics (Poincaré, Sample Entropy, DFA)
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_hrv_nonlinear(rr: np.ndarray) -> dict:
        """
        Compute non-linear HRV metrics: Poincaré SD1/SD2, Sample Entropy, DFA α1.

        Parameters
        ----------
        rr : 1-D array of RR intervals in ms

        Returns
        -------
        dict with keys: sd1, sd2, sd1_sd2_ratio, sample_entropy, dfa_alpha1
        """
        diff_rr = np.diff(rr)
        sd1 = float(np.std(diff_rr) / np.sqrt(2))
        sd2 = float(np.sqrt(max(0.0, 2 * np.std(rr) ** 2 - sd1 ** 2)))
        sd1_sd2_ratio = sd1 / sd2 if sd2 > 0 else None

        # Sample Entropy approximation (no extra dependencies)
        try:
            def _sample_entropy(ts, m=2, r_tol=0.2):
                r = r_tol * np.std(ts)
                N = len(ts)

                def count_matches(template, ts, r):
                    count = 0
                    for i in range(len(ts) - len(template)):
                        if np.max(np.abs(ts[i:i + len(template)] - template)) < r:
                            count += 1
                    return count

                B = sum(count_matches(ts[i:i + m], ts, r) for i in range(N - m)) / (N - m)
                A = sum(count_matches(ts[i:i + m + 1], ts, r) for i in range(N - m - 1)) / (N - m - 1)
                return -np.log(A / B) if B > 0 and A > 0 else None

            sampen = _sample_entropy(rr) if len(rr) >= 20 else None
        except Exception:
            sampen = None

        # DFA α1 (short-range, scale 4–16)
        try:
            def _dfa(ts, scale_min=4, scale_max=16):
                ts = ts - np.mean(ts)
                cumsum = np.cumsum(ts)
                scales = np.unique(
                    np.logspace(np.log10(scale_min), np.log10(scale_max), 8).astype(int)
                )
                flucts = []
                for n in scales:
                    n_segments = len(ts) // n
                    if n_segments < 2:
                        continue
                    rms_list = []
                    for seg in range(n_segments):
                        seg_data = cumsum[seg * n:(seg + 1) * n]
                        x = np.arange(len(seg_data))
                        fit = np.polyfit(x, seg_data, 1)
                        trend = np.polyval(fit, x)
                        rms_list.append(np.sqrt(np.mean((seg_data - trend) ** 2)))
                    flucts.append(np.mean(rms_list))
                if len(flucts) >= 2:
                    log_scales = np.log10(scales[:len(flucts)])
                    log_flucts = np.log10(flucts)
                    alfa1 = float(np.polyfit(log_scales, log_flucts, 1)[0])
                    return alfa1
                return None

            alfa1 = _dfa(rr) if len(rr) >= 20 else None
        except Exception:
            alfa1 = None

        return {
            "sd1": sd1,
            "sd2": sd2,
            "sd1_sd2_ratio": sd1_sd2_ratio,
            "sample_entropy": sampen,
            "dfa_alpha1": alfa1,
        }

    # ------------------------------------------------------------------
    # Arrhythmia detection (AFib suspicion, ectopic beats, HR class)
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_arrhythmia(rr: np.ndarray, peaks: np.ndarray, fs: int) -> dict:
        """
        Lightweight rule-based arrhythmia screening from RR intervals.

        Parameters
        ----------
        rr    : 1-D array of RR intervals in ms (already artefact-filtered)
        peaks : R-peak sample indices
        fs    : sampling rate in Hz

        Returns
        -------
        dict with keys: afib_suspected, afib_evidence, ectopic_beats,
                        ectopic_ratio, heart_rate_class, rr_cv
        """
        results = {
            "afib_suspected": False,
            "afib_evidence": [],
            "ectopic_beats": 0,
            "ectopic_ratio": 0.0,
            "heart_rate_class": "normal",
            "rr_cv": 0.0,
        }
        if len(rr) < 5:
            return results

        mean_rr = np.mean(rr)
        std_rr = np.std(rr)
        cv = float(std_rr / mean_rr) if mean_rr > 0 else 0.0
        results["rr_cv"] = round(cv, 4)

        mean_hr = 60000.0 / mean_rr if mean_rr > 0 else 0.0
        if mean_hr < 50:
            results["heart_rate_class"] = "bradycardia"
        elif mean_hr > 100:
            results["heart_rate_class"] = "tachycardia"
        else:
            results["heart_rate_class"] = "normal"

        # AFib: high RR irregularity + elevated RMSSD/meanRR
        afib_evidence = []
        if cv > 0.20:
            afib_evidence.append(
                f"High RR coefficient of variation: {cv:.3f} (>0.20)"
            )
        rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
        if rmssd / mean_rr > 0.15 and cv > 0.15:
            afib_evidence.append(
                f"Irregularity index elevated: RMSSD/meanRR={rmssd / mean_rr:.3f}"
            )
        results["afib_suspected"] = len(afib_evidence) >= 2
        results["afib_evidence"] = afib_evidence

        # Ectopic beats: RR < 80 % or > 130 % of preceding interval
        ectopic = 0
        for i in range(1, len(rr)):
            ratio = rr[i] / rr[i - 1]
            if ratio < 0.80 or ratio > 1.30:
                ectopic += 1
        results["ectopic_beats"] = ectopic
        results["ectopic_ratio"] = round(ectopic / len(rr), 4)

        return results

    # ------------------------------------------------------------------
    # Main orchestrator
    # ------------------------------------------------------------------
    @staticmethod
    def analyze(
        signal: np.ndarray,
        fs: int,
        config: PreprocessingConfig,
        compute_hrv: bool = True,
        compute_morphology: bool = True,
        compute_nonlinear: bool = True,
        compute_arrhythmia: bool = True,
    ) -> dict:
        """
        Full ECG analysis pipeline.

        Parameters
        ----------
        signal            : preprocessed 1-D ECG array
        fs                : sampling rate in Hz
        config            : PreprocessingConfig (used for context only here)
        compute_hrv       : whether to compute HRV metrics
        compute_morphology: whether to run wave delineation

        Returns
        -------
        dict with keys:
            n_peaks, rr_intervals_ms,
            hrv_time (dict), hrv_freq (dict),
            morphology (dict),
            hrv_nonlinear (dict),
            arrhythmia (dict),
            warnings (list[str])
        """
        result: dict = {
            "n_peaks": 0,
            "rr_intervals_ms": [],
            "hrv_time": None,
            "hrv_freq": None,
            "morphology": None,
            "hrv_nonlinear": None,
            "arrhythmia": None,
            "warnings": [],
        }

        # ---- R-peak detection -----------------------------------------
        try:
            r_peaks = ECGAnalyzer.detect_peaks(signal, fs)
        except ValueError as exc:
            result["warnings"].append(f"Peak detection error: {exc}")
            return result

        result["n_peaks"] = int(len(r_peaks))

        if len(r_peaks) < 2:
            result["warnings"].append(
                f"Only {len(r_peaks)} R-peak(s) detected — insufficient for analysis. "
                "Check signal quality and sampling rate."
            )
            return result

        # RR intervals in ms
        rr_ms = np.diff(r_peaks.astype(np.float64)) / fs * 1000.0

        # Artefact rejection: discard physiologically implausible RR intervals
        # Normal HR range 20–300 bpm → RR 200–3000 ms
        valid_mask = (rr_ms >= 200.0) & (rr_ms <= 3000.0)
        n_rejected = int((~valid_mask).sum())
        if n_rejected > 0:
            result["warnings"].append(
                f"{n_rejected} RR interval(s) outside 200–3000 ms were discarded "
                "as artefacts."
            )
        rr_ms = rr_ms[valid_mask]
        result["rr_intervals_ms"] = rr_ms.tolist()

        if rr_ms.size < 2:
            result["warnings"].append(
                "Fewer than 2 valid RR intervals after artefact rejection — "
                "HRV computation skipped."
            )
            return result

        # ---- HRV time domain ------------------------------------------
        if compute_hrv:
            try:
                result["hrv_time"] = ECGAnalyzer.compute_hrv_time(rr_ms)
            except Exception as exc:
                result["warnings"].append(f"HRV time-domain failed: {exc}")

            try:
                result["hrv_freq"] = ECGAnalyzer.compute_hrv_freq(rr_ms)
            except Exception as exc:
                result["warnings"].append(f"HRV frequency-domain failed: {exc}")

        # ---- Morphology -----------------------------------------------
        if compute_morphology:
            try:
                result["morphology"] = ECGAnalyzer.extract_morphology(
                    signal, fs, r_peaks
                )
            except Exception as exc:
                result["warnings"].append(f"Morphology extraction failed: {exc}")
                result["morphology"] = {
                    "pr_interval_ms": None,
                    "qrs_duration_ms": None,
                    "qtc_ms": None,
                    "st_deviation_mv": None,
                }

        # ---- HRV non-linear -------------------------------------------
        if compute_nonlinear and rr_ms.size >= 2:
            try:
                result["hrv_nonlinear"] = ECGAnalyzer._compute_hrv_nonlinear(rr_ms)
            except Exception as exc:
                result["warnings"].append(f"HRV non-linear computation failed: {exc}")

        # ---- Arrhythmia detection -------------------------------------
        if compute_arrhythmia and rr_ms.size >= 2:
            try:
                result["arrhythmia"] = ECGAnalyzer._detect_arrhythmia(
                    rr_ms, r_peaks, fs
                )
            except Exception as exc:
                result["warnings"].append(f"Arrhythmia detection failed: {exc}")

        return result

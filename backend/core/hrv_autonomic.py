"""
Autonomic nervous system indices — PNS, SNS, Baevsky Stress Index.
Based on letteratura HRV (Shaffer & Ginsberg 2017) and Shaffer & Ginsberg (2017).
"""
import numpy as np
from typing import Optional


class AutonomicIndexCalculator:

    # Normative reference values (Shaffer & Ginsberg 2017, healthy adults at rest)
    _NORMS = {
        "mean_rr_ms": (926.0, 90.0),   # mean, std
        "rmssd_ms":   (42.0,  15.0),
        "sd1_ms":     (29.0,  10.0),
        "mean_hr_bpm":(65.0,  8.0),
        "lf_hf_ratio":(2.0,   1.2),
        "sd2_ms":     (40.0,  14.0),
    }

    @classmethod
    def compute_pns_index(cls,
                          mean_rr_ms: float,
                          rmssd_ms: float,
                          sd1_ms: float) -> Optional[float]:
        """
        PNS Index — parasympathetic nervous system activity.
        Formula (HRV avanzato): mean of z-scores of (mean_RR, RMSSD, SD1) using reference population.
        Range: -2 to +2. Positive = high vagal/parasympathetic activity (resting, recovered).
        """
        try:
            mu_rr, s_rr = cls._NORMS["mean_rr_ms"]
            mu_rmssd, s_rmssd = cls._NORMS["rmssd_ms"]
            mu_sd1, s_sd1 = cls._NORMS["sd1_ms"]
            z_rr = (mean_rr_ms - mu_rr) / s_rr
            z_rmssd = (rmssd_ms - mu_rmssd) / s_rmssd
            z_sd1 = (sd1_ms - mu_sd1) / s_sd1
            return round(float((z_rr + z_rmssd + z_sd1) / 3.0), 3)
        except Exception:
            return None

    @classmethod
    def compute_sns_index(cls,
                          mean_hr_bpm: float,
                          lf_hf_ratio: Optional[float],
                          sd2_ms: float) -> Optional[float]:
        """
        SNS Index — sympathetic nervous system activity.
        Formula (HRV avanzato): mean of z-scores of (mean_HR, LF/HF, SD2).
        Range: -2 to +2. Positive = high sympathetic activity (stress, exercise).
        """
        try:
            mu_hr, s_hr = cls._NORMS["mean_hr_bpm"]
            mu_lfhf, s_lfhf = cls._NORMS["lf_hf_ratio"]
            mu_sd2, s_sd2 = cls._NORMS["sd2_ms"]
            z_hr = (mean_hr_bpm - mu_hr) / s_hr
            z_lfhf = ((lf_hf_ratio or mu_lfhf) - mu_lfhf) / s_lfhf
            z_sd2 = (sd2_ms - mu_sd2) / s_sd2
            return round(float((z_hr + z_lfhf + z_sd2) / 3.0), 3)
        except Exception:
            return None

    @staticmethod
    def compute_baevsky_stress_index(rr: np.ndarray) -> Optional[float]:
        """
        Baevsky Stress Index (SI).
        SI = AMo / (2 × Mo × MxDMn)
        where:
          Mo = mode of RR histogram (most frequent value, in seconds)
          AMo = amplitude of mode (% of beats in modal bin)
          MxDMn = variation range = (max_RR - min_RR) / 1000 [seconds]
        Normal range: 50-150. >150 = stress/sympathetic dominance.
        """
        if len(rr) < 10:
            return None
        try:
            rr_s = rr / 1000.0  # convert to seconds
            bin_width = 0.05  # 50ms bins
            bins = np.arange(np.min(rr_s) - bin_width, np.max(rr_s) + bin_width, bin_width)
            hist, edges = np.histogram(rr_s, bins=bins)
            if len(hist) == 0 or np.max(hist) == 0:
                return None
            peak_idx = int(np.argmax(hist))
            mo = float((edges[peak_idx] + edges[peak_idx + 1]) / 2.0)  # modal value in seconds
            amo = float(hist[peak_idx] / len(rr) * 100.0)              # amplitude in %
            mxdmn = float((np.max(rr_s) - np.min(rr_s)))              # variation range in seconds
            if mo <= 0 or mxdmn <= 0:
                return None
            si = amo / (2.0 * mo * mxdmn)
            return round(float(si), 2)
        except Exception:
            return None

    @classmethod
    def compute_autonomic_balance(cls, pns_index: Optional[float],
                                  sns_index: Optional[float]) -> Optional[float]:
        """
        Autonomic Balance = PNS - SNS.
        Positive: parasympathetic dominant (recovery, rest).
        Negative: sympathetic dominant (stress, exercise).
        """
        if pns_index is None or sns_index is None:
            return None
        return round(float(pns_index - sns_index), 3)

    @classmethod
    def analyze(cls, rr: np.ndarray, hrv_time: dict, hrv_nonlinear: dict,
                hrv_freq: dict) -> dict:
        """Compute all autonomic indices from pre-computed HRV dicts."""
        mean_rr = float(np.mean(rr)) if len(rr) > 0 else 60000.0 / hrv_time.get("mean_hr", 70)
        rmssd = hrv_time.get("rmssd") or 0.0
        sd1 = hrv_nonlinear.get("sd1") or 0.0
        sd2 = hrv_nonlinear.get("sd2") or 0.0
        mean_hr = hrv_time.get("mean_hr") or 70.0
        lf_hf = hrv_freq.get("lf_hf_ratio")

        pns = cls.compute_pns_index(mean_rr, rmssd, sd1)
        sns = cls.compute_sns_index(mean_hr, lf_hf, sd2)
        baevsky = cls.compute_baevsky_stress_index(rr)
        balance = cls.compute_autonomic_balance(pns, sns)

        return {
            "pns_index": pns,
            "sns_index": sns,
            "baevsky_stress_index": baevsky,
            "autonomic_balance": balance,
        }

"""
RR interval artifact detection and correction — algoritmo adattivo pipeline.
Detects ectopic beats, missing beats, and signal dropouts in RR series.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from typing import Optional


class ArtifactCorrector:

    @staticmethod
    def detect_artifacts(rr: np.ndarray, method: str = "combined") -> np.ndarray:
        """
        Detect artifact indices in RR series.
        Returns boolean mask: True = artifact.

        Methods:
          "threshold"      — physiological limits only
          "quotient"       — local ratio filter
          "moving_median"  — adaptive median filter
          "combined"       — all three (default)
        """
        rr = np.asarray(rr, dtype=float)
        if len(rr) < 3:
            return np.zeros(len(rr), dtype=bool)

        artifacts = np.zeros(len(rr), dtype=bool)

        if method in ("threshold", "combined"):
            # Physiological limits: 200-2000 ms (HR 30-300 bpm)
            artifacts |= (rr < 200) | (rr > 2000)

        if method in ("quotient", "combined"):
            # Quotient filter: flag if |RR(i)/RR(i-1) - 1| > 0.2
            ratios = np.abs(np.diff(rr) / (rr[:-1] + 1e-9))
            quotient_flags = np.zeros(len(rr), dtype=bool)
            quotient_flags[1:] = ratios > 0.20
            artifacts |= quotient_flags

        if method in ("moving_median", "combined"):
            # Moving median filter: flag if |RR(i) - median_local| > 25% of median
            window = 5
            half = window // 2
            med_flags = np.zeros(len(rr), dtype=bool)
            for i in range(len(rr)):
                start = max(0, i - half)
                end = min(len(rr), i + half + 1)
                local_med = np.median(rr[start:end])
                if local_med > 0 and abs(rr[i] - local_med) / local_med > 0.25:
                    med_flags[i] = True
            artifacts |= med_flags

        return artifacts

    @staticmethod
    def correct_artifacts(rr: np.ndarray, artifact_mask: np.ndarray,
                          method: str = "cubic_spline") -> np.ndarray:
        """
        Interpolate over detected artifacts.

        Methods: "linear", "cubic_spline", "moving_average", "delete"
        Returns corrected RR series (same length unless method="delete").
        """
        rr = np.asarray(rr, dtype=float)
        artifact_mask = np.asarray(artifact_mask, dtype=bool)
        if not np.any(artifact_mask):
            return rr.copy()

        rr_corrected = rr.copy()
        artifact_idx = np.where(artifact_mask)[0]
        valid_idx = np.where(~artifact_mask)[0]

        if len(valid_idx) < 2:
            return rr_corrected  # not enough valid points

        if method == "delete":
            return rr[~artifact_mask]

        if method == "linear":
            rr_corrected[artifact_idx] = np.interp(
                artifact_idx, valid_idx, rr[valid_idx]
            )

        elif method == "cubic_spline":
            try:
                cs = CubicSpline(valid_idx, rr[valid_idx], extrapolate=True)
                rr_corrected[artifact_idx] = cs(artifact_idx)
                # Clamp to physiological range
                rr_corrected = np.clip(rr_corrected, 200, 2000)
            except Exception:
                # Fallback to linear
                rr_corrected[artifact_idx] = np.interp(
                    artifact_idx, valid_idx, rr[valid_idx]
                )

        elif method == "moving_average":
            window = 5
            half = window // 2
            for i in artifact_idx:
                start = max(0, i - half)
                end = min(len(rr), i + half + 1)
                valid_local = [rr[j] for j in range(start, end)
                               if not artifact_mask[j]]
                if valid_local:
                    rr_corrected[i] = float(np.mean(valid_local))

        return rr_corrected

    @classmethod
    def compute_artifact_stats(cls, rr: np.ndarray,
                                artifact_mask: np.ndarray) -> dict:
        """Summary statistics about detected artifacts."""
        rr = np.asarray(rr, dtype=float)
        artifact_mask = np.asarray(artifact_mask, dtype=bool)
        n_total = len(rr)
        n_artifacts = int(np.sum(artifact_mask))
        artifact_ratio = float(n_artifacts / n_total) if n_total > 0 else 0.0

        quality = "Good"
        if artifact_ratio > 0.20:
            quality = "Poor"
        elif artifact_ratio > 0.05:
            quality = "Acceptable"

        return {
            "n_total_beats": n_total,
            "n_artifacts": n_artifacts,
            "artifact_ratio": round(artifact_ratio, 4),
            "artifact_pct": round(artifact_ratio * 100, 2),
            "quality_label": quality,
            "correctable": artifact_ratio < 0.20,
        }

    @classmethod
    def process(cls, rr: np.ndarray,
                detection_method: str = "combined",
                correction_method: str = "cubic_spline") -> dict:
        """
        Full pipeline: detect → correct → report stats.
        Returns: corrected_rr, artifact_mask, stats
        """
        rr = np.asarray(rr, dtype=float)
        artifact_mask = cls.detect_artifacts(rr, method=detection_method)
        corrected_rr = cls.correct_artifacts(rr, artifact_mask, method=correction_method)
        stats = cls.compute_artifact_stats(rr, artifact_mask)

        return {
            "corrected_rr": corrected_rr,
            "artifact_mask": artifact_mask,
            "stats": stats,
        }

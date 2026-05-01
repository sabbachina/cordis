"""
Advanced HRV nonlinear analysis — analisi HRV scientifica avanzata.
All methods accept rr: np.ndarray (RR intervals in milliseconds).
"""
import numpy as np
from typing import Optional


class HRVAdvancedAnalyzer:

    @staticmethod
    def compute_dfa_alpha2(rr: np.ndarray, scale_min: int = 16, scale_max: int = 64) -> Optional[float]:
        """DFA long-range scaling exponent α2 (scale 16-64 beats). Normal: 0.9-1.1"""
        # Stesso algoritmo di dfa_alpha1 ma su scale più lunghe
        # Usa lo stesso _dfa helper già presente in ecg_analyzer ma riscrivilo qui standalone
        if len(rr) < scale_max * 2:
            return None
        try:
            ts = rr - np.mean(rr)
            cumsum = np.cumsum(ts)
            scales = np.unique(np.logspace(np.log10(scale_min), np.log10(scale_max), 10).astype(int))
            flucts = []
            valid_scales = []
            for n in scales:
                n_segs = len(ts) // n
                if n_segs < 2:
                    continue
                rms_list = []
                for seg_i in range(n_segs):
                    seg = cumsum[seg_i*n:(seg_i+1)*n]
                    x = np.arange(len(seg))
                    fit = np.polyfit(x, seg, 1)
                    trend = np.polyval(fit, x)
                    rms_list.append(float(np.sqrt(np.mean((seg - trend)**2))))
                flucts.append(float(np.mean(rms_list)))
                valid_scales.append(n)
            if len(flucts) < 2:
                return None
            log_s = np.log10(valid_scales)
            log_f = np.log10(np.array(flucts))
            return float(np.polyfit(log_s, log_f, 1)[0])
        except Exception:
            return None

    @staticmethod
    def compute_approximate_entropy(rr: np.ndarray, m: int = 2, r_tol: float = 0.2) -> Optional[float]:
        """ApEn — Approximate Entropy. Normal: 0.7-1.5. Lower = more regular."""
        if len(rr) < 10:
            return None
        try:
            r = r_tol * float(np.std(rr))
            if r == 0:
                return None
            N = len(rr)

            def _phi(m_len):
                count = 0
                for i in range(N - m_len + 1):
                    template = rr[i:i + m_len]
                    matches = 0
                    for j in range(N - m_len + 1):
                        if np.max(np.abs(rr[j:j + m_len] - template)) <= r:
                            matches += 1
                    if matches > 0:
                        count += np.log(matches / (N - m_len + 1))
                return count / (N - m_len + 1)

            return float(_phi(m) - _phi(m + 1))
        except Exception:
            return None

    @staticmethod
    def compute_fuzzy_entropy(rr: np.ndarray, m: int = 2, r_tol: float = 0.2, n: int = 2) -> Optional[float]:
        """FuzzyEn — Fuzzy Entropy. More robust than SampEn for short series."""
        if len(rr) < 10:
            return None
        try:
            r = r_tol * float(np.std(rr))
            if r == 0:
                return None
            N = len(rr)
            rr_centered = rr - np.mean(rr)  # remove baseline

            def _fuzzy_count(m_len):
                total = 0.0
                for i in range(N - m_len):
                    xi = rr_centered[i:i + m_len]
                    xi = xi - np.mean(xi)  # remove local mean
                    count = 0.0
                    for j in range(N - m_len):
                        if i == j:
                            continue
                        xj = rr_centered[j:j + m_len]
                        xj = xj - np.mean(xj)
                        d = np.max(np.abs(xi - xj))
                        # Fuzzy membership function: exp(-(d/r)^n)
                        count += np.exp(-(d / r) ** n)
                    total += count / (N - m_len - 1)
                return total / (N - m_len)

            phi_m = _fuzzy_count(m)
            phi_m1 = _fuzzy_count(m + 1)
            if phi_m <= 0 or phi_m1 <= 0:
                return None
            return float(-np.log(phi_m1 / phi_m))
        except Exception:
            return None

    @staticmethod
    def compute_multiscale_entropy(rr: np.ndarray, max_scale: int = 10) -> Optional[list]:
        """MSE — Multiscale Entropy. Returns list of SampEn at scales 1..max_scale."""
        # Limit max_scale based on signal length
        max_scale = min(max_scale, len(rr) // 10)
        if max_scale < 2:
            return None
        try:
            def _coarse_grain(ts, scale):
                n = len(ts) // scale
                return np.array([np.mean(ts[i*scale:(i+1)*scale]) for i in range(n)])

            def _sample_entropy_fast(ts, m=2, r_tol=0.15):
                r = r_tol * np.std(ts)
                if r == 0:
                    return None
                N = len(ts)
                if N < m + 2:
                    return None
                # Vectorized approximation using sliding windows
                B, A = 0, 0
                for i in range(N - m - 1):
                    template_m = ts[i:i+m]
                    template_m1 = ts[i:i+m+1]
                    for j in range(i+1, N - m):
                        if np.max(np.abs(ts[j:j+m] - template_m)) < r:
                            B += 1
                            if j < N - m and np.max(np.abs(ts[j:j+m+1] - template_m1)) < r:
                                A += 1
                if B == 0:
                    return None
                return float(-np.log(A / B)) if A > 0 else None

            mse_values = []
            for scale in range(1, max_scale + 1):
                cg = _coarse_grain(rr, scale)
                se = _sample_entropy_fast(cg)
                mse_values.append(se)
            return mse_values
        except Exception:
            return None

    @staticmethod
    def compute_rqa(rr: np.ndarray, dim: int = 10, tau: int = 1, threshold_pct: float = 0.20) -> dict:
        """
        Recurrence Quantification Analysis (RQA).
        Returns: rr_pct, det, avg_diag_length, max_diag_length, entr
        Normal ranges (approximate): RR%: 1-5%, DET: 50-90%, L: 2-5, ENTR: 0.5-2.5
        """
        result = {"rr_pct": None, "det": None, "avg_diag_length": None,
                  "max_diag_length": None, "entr": None}
        if len(rr) < dim * tau + 5:
            return result
        try:
            # Phase space reconstruction (time-delay embedding)
            N = len(rr) - (dim - 1) * tau
            if N < 5:
                return result
            embedded = np.array([rr[i:i + dim * tau:tau] for i in range(N)])

            # Distance matrix
            threshold = threshold_pct * np.std(rr)
            dist = np.max(np.abs(embedded[:, None, :] - embedded[None, :, :]), axis=2)
            recurrence = (dist < threshold).astype(np.uint8)
            np.fill_diagonal(recurrence, 0)  # remove line of identity

            total_points = N * N - N
            rec_points = int(np.sum(recurrence))
            rr_pct = float(rec_points / total_points * 100) if total_points > 0 else 0.0
            result["rr_pct"] = round(rr_pct, 3)

            # Diagonal lines analysis (DET, L, Lmax, ENTR)
            diag_lengths = []
            for offset in range(1, N):
                diag = np.diagonal(recurrence, offset)
                count = 0
                for val in diag:
                    if val:
                        count += 1
                    elif count >= 2:
                        diag_lengths.append(count)
                        count = 0
                if count >= 2:
                    diag_lengths.append(count)
                # Also check negative diagonal
                diag_neg = np.diagonal(recurrence, -offset)
                count = 0
                for val in diag_neg:
                    if val:
                        count += 1
                    elif count >= 2:
                        diag_lengths.append(count)
                        count = 0
                if count >= 2:
                    diag_lengths.append(count)

            if diag_lengths:
                diag_arr = np.array(diag_lengths)
                det = float(np.sum(diag_arr) / rec_points * 100) if rec_points > 0 else 0.0
                result["det"] = round(det, 3)
                result["avg_diag_length"] = round(float(np.mean(diag_arr)), 3)
                result["max_diag_length"] = int(np.max(diag_arr))
                # Shannon entropy of diagonal length distribution
                lengths, counts = np.unique(diag_arr, return_counts=True)
                probs = counts / counts.sum()
                entr = float(-np.sum(probs * np.log2(probs + 1e-12)))
                result["entr"] = round(entr, 3)

            return result
        except Exception:
            return result

    @staticmethod
    def compute_lyapunov(rr: np.ndarray, emb_dim: int = 10, tau: int = 1) -> Optional[float]:
        """
        Largest Lyapunov Exponent (LLE) via Rosenstein method.
        Positive LLE → chaotic (healthy HRV). Normal: 0.005-0.02 (in ms⁻¹ units).
        """
        if len(rr) < emb_dim * tau + 20:
            return None
        try:
            N = len(rr) - (emb_dim - 1) * tau
            if N < 10:
                return None
            embedded = np.array([rr[i:i + emb_dim * tau:tau] for i in range(N)])

            # For each point find nearest neighbor (excluding temporal neighbors)
            min_temporal_sep = int(1.0 / (np.mean(rr) / 1000.0))  # ~1 beat
            divergence = []
            for i in range(N):
                dists = np.max(np.abs(embedded - embedded[i]), axis=1)
                dists[max(0, i - min_temporal_sep):min(N, i + min_temporal_sep + 1)] = np.inf
                nn_idx = int(np.argmin(dists))
                if dists[nn_idx] == np.inf:
                    continue
                # Track divergence over time
                d_list = []
                for step in range(1, min(20, N - max(i, nn_idx))):
                    if i + step < N and nn_idx + step < N:
                        d = np.max(np.abs(embedded[i + step] - embedded[nn_idx + step]))
                        if d > 0:
                            d_list.append(np.log(d))
                if d_list:
                    divergence.append(d_list)

            if len(divergence) < 5:
                return None
            # Average log-divergence curve
            max_steps = min(len(d) for d in divergence)
            avg_div = np.array([np.mean([d[t] for d in divergence if len(d) > t])
                                for t in range(max_steps)])
            # LLE = slope of linear fit of avg divergence
            steps = np.arange(len(avg_div))
            if len(steps) < 3:
                return None
            lle = float(np.polyfit(steps, avg_div, 1)[0])
            return round(lle, 6)
        except Exception:
            return None

    @classmethod
    def analyze_all(cls, rr: np.ndarray) -> dict:
        """Run all advanced analyses. Returns dict with all computed values."""
        warnings = []
        result = {}

        result["dfa_alpha2"] = cls.compute_dfa_alpha2(rr)
        if result["dfa_alpha2"] is None:
            warnings.append("DFA α2 requires ≥128 RR intervals")

        result["approximate_entropy"] = cls.compute_approximate_entropy(rr)
        result["fuzzy_entropy"] = cls.compute_fuzzy_entropy(rr)

        mse = cls.compute_multiscale_entropy(rr)
        result["mse_values"] = mse
        result["mse_slope"] = None
        if mse and len([v for v in mse if v is not None]) >= 3:
            valid = [(i+1, v) for i, v in enumerate(mse) if v is not None]
            scales, values = zip(*valid)
            result["mse_slope"] = float(np.polyfit(scales, values, 1)[0])

        rqa = cls.compute_rqa(rr)
        result.update(rqa)  # rr_pct, det, avg_diag_length, max_diag_length, entr

        result["lyapunov_exponent"] = cls.compute_lyapunov(rr)

        result["warnings"] = warnings
        return result

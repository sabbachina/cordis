"""
Test suite for FastAPI endpoints using TestClient.

Covers:
  - GET /health
  - GET /signals/health
  - GET /signals/sample/ECG
  - GET /signals/sample/PPG
  - POST /analysis/analyze  (ECG full pipeline)
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ecg_payload(duration: int = 30, fs: int = 500) -> dict:
    """Build a minimal AnalysisRequest payload with a synthetic ECG signal."""
    import neurokit2 as nk
    sig = nk.ecg_simulate(duration=duration, sampling_rate=fs, noise=0.02)
    return {
        "signal": {
            "signal_type": "ECG",
            "sampling_rate": fs,
            "values": list(sig),
        },
        "preprocessing": {
            "lowcut": 0.5,
            "highcut": 40.0,
            "method": "butterworth",
            "remove_baseline": True,
        },
        "compute_hrv": True,
        "compute_morphology": True,
        "compute_ml": False,   # skip ML to keep test fast and dependency-free
    }


def _ppg_payload(duration: int = 30, fs: int = 125) -> dict:
    """Build a minimal AnalysisRequest payload with a synthetic PPG signal."""
    import neurokit2 as nk
    sig = nk.ppg_simulate(duration=duration, sampling_rate=fs, heart_rate=70)
    return {
        "signal": {
            "signal_type": "PPG",
            "sampling_rate": fs,
            "values": list(sig),
        },
        "preprocessing": {
            "lowcut": 0.5,
            "highcut": 10.0,
            "method": "butterworth",
            "remove_baseline": True,
        },
        "compute_hrv": True,
        "compute_morphology": False,
        "compute_ml": False,
    }


# ---------------------------------------------------------------------------
# Test 1 – GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_status_code_200(self):
        response = client.get("/health")
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}"
        )

    def test_body_contains_status_ok(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data, "Response must contain 'status' key"
        assert data["status"] == "ok", f"Expected status='ok', got {data['status']}"

    def test_content_type_is_json(self):
        response = client.get("/health")
        assert "application/json" in response.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# Test 2 – GET /signals/health
# ---------------------------------------------------------------------------

class TestSignalsHealthEndpoint:
    def test_status_code_200(self):
        response = client.get("/signals/health")
        assert response.status_code == 200

    def test_returns_json(self):
        response = client.get("/signals/health")
        data = response.json()
        assert isinstance(data, dict)

    def test_has_status_key(self):
        response = client.get("/signals/health")
        data = response.json()
        assert "status" in data


# ---------------------------------------------------------------------------
# Test 3 – GET /signals/sample/ECG
# ---------------------------------------------------------------------------

class TestSampleECG:
    def test_status_code_200(self):
        response = client.get("/signals/sample/ECG")
        assert response.status_code == 200

    def test_returns_values_list(self):
        response = client.get("/signals/sample/ECG")
        data = response.json()
        assert "values" in data, "Response must contain 'values' key"
        assert isinstance(data["values"], list), "'values' must be a list"

    def test_values_list_is_nonempty(self):
        response = client.get("/signals/sample/ECG")
        data = response.json()
        assert len(data["values"]) > 0, "'values' must not be empty"

    def test_sampling_rate_present(self):
        response = client.get("/signals/sample/ECG")
        data = response.json()
        assert "sampling_rate" in data
        assert data["sampling_rate"] == 500

    def test_n_samples_matches_values_length(self):
        response = client.get("/signals/sample/ECG")
        data = response.json()
        assert data["n_samples"] == len(data["values"])

    def test_signal_type_is_ecg(self):
        response = client.get("/signals/sample/ECG")
        data = response.json()
        assert data["signal_type"] == "ECG"


# ---------------------------------------------------------------------------
# Test 4 – GET /signals/sample/PPG
# ---------------------------------------------------------------------------

class TestSamplePPG:
    def test_status_code_200(self):
        response = client.get("/signals/sample/PPG")
        assert response.status_code == 200

    def test_returns_nonempty_values(self):
        response = client.get("/signals/sample/PPG")
        data = response.json()
        assert "values" in data
        assert len(data["values"]) > 0

    def test_sampling_rate_125(self):
        response = client.get("/signals/sample/PPG")
        data = response.json()
        assert data["sampling_rate"] == 125

    def test_signal_type_is_ppg(self):
        response = client.get("/signals/sample/PPG")
        data = response.json()
        assert data["signal_type"] == "PPG"

    def test_invalid_type_returns_400(self):
        response = client.get("/signals/sample/INVALID")
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# Test 5 – POST /analysis/analyze  (ECG full pipeline)
# ---------------------------------------------------------------------------

class TestAnalyzeEndpointECG:
    def test_status_code_200(self):
        payload = _ecg_payload()
        response = client.post("/analysis/analyze", json=payload)
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}. "
            f"Detail: {response.text[:300]}"
        )

    def test_response_is_biomarker_report(self):
        payload = _ecg_payload()
        response = client.post("/analysis/analyze", json=payload)
        data = response.json()
        assert isinstance(data, dict)
        # BiomarkerReport required fields
        for field in ("signal_type", "duration_seconds", "sampling_rate",
                      "n_peaks_detected", "warnings"):
            assert field in data, f"Missing field in BiomarkerReport: {field}"

    def test_hrv_time_not_none(self):
        """30-second ECG should always produce hrv_time results."""
        payload = _ecg_payload()
        response = client.post("/analysis/analyze", json=payload)
        data = response.json()
        assert data.get("hrv_time") is not None, (
            "hrv_time should not be None for 30-second ECG input"
        )

    def test_hrv_time_has_mean_hr(self):
        payload = _ecg_payload()
        response = client.post("/analysis/analyze", json=payload)
        data = response.json()
        hrv_time = data.get("hrv_time")
        if hrv_time is not None:
            assert "mean_hr" in hrv_time
            mean_hr = hrv_time["mean_hr"]
            if isinstance(mean_hr, dict):
                # BiomarkerValue has a 'value' field
                assert mean_hr.get("value") is not None

    def test_signal_type_echoed(self):
        payload = _ecg_payload()
        response = client.post("/analysis/analyze", json=payload)
        data = response.json()
        assert data.get("signal_type") == "ECG"

    def test_warnings_is_list(self):
        payload = _ecg_payload()
        response = client.post("/analysis/analyze", json=payload)
        data = response.json()
        assert isinstance(data.get("warnings"), list)

    def test_n_peaks_positive(self):
        payload = _ecg_payload()
        response = client.post("/analysis/analyze", json=payload)
        data = response.json()
        assert data.get("n_peaks_detected", 0) > 0


# ---------------------------------------------------------------------------
# Test 6 – POST /analysis/analyze  (PPG pipeline)
# ---------------------------------------------------------------------------

class TestAnalyzeEndpointPPG:
    def test_status_code_200(self):
        payload = _ppg_payload()
        response = client.post("/analysis/analyze", json=payload)
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}. "
            f"Detail: {response.text[:300]}"
        )

    def test_signal_type_ppg(self):
        payload = _ppg_payload()
        response = client.post("/analysis/analyze", json=payload)
        data = response.json()
        assert data.get("signal_type") == "PPG"

    def test_n_peaks_positive(self):
        payload = _ppg_payload()
        response = client.post("/analysis/analyze", json=payload)
        data = response.json()
        assert data.get("n_peaks_detected", 0) > 0


# ---------------------------------------------------------------------------
# Test 7 – Error handling
# ---------------------------------------------------------------------------

class TestAnalyzeEndpointErrors:
    def test_too_short_signal_returns_error(self):
        """Sending a signal shorter than the min_length=100 should fail validation."""
        payload = {
            "signal": {
                "signal_type": "ECG",
                "sampling_rate": 500,
                "values": [0.1] * 50,   # too short: < 100 samples
            },
            "compute_hrv": True,
            "compute_morphology": False,
            "compute_ml": False,
        }
        response = client.post("/analysis/analyze", json=payload)
        # Pydantic validation error → 422 Unprocessable Entity
        assert response.status_code == 422

    def test_invalid_sampling_rate_returns_error(self):
        """Sampling rate < 50 Hz should fail Pydantic validation."""
        import neurokit2 as nk
        sig = nk.ecg_simulate(duration=10, sampling_rate=500, noise=0.02)
        payload = {
            "signal": {
                "signal_type": "ECG",
                "sampling_rate": 10,   # below minimum of 50
                "values": list(sig),
            },
            "compute_hrv": False,
            "compute_morphology": False,
            "compute_ml": False,
        }
        response = client.post("/analysis/analyze", json=payload)
        assert response.status_code == 422

import httpx
import os
import numpy as np

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def get_sample_signal(signal_type: str) -> dict:
    """Scarica segnale campione dal backend."""
    with httpx.Client(timeout=30.0) as client:
        response = client.get(f"{BACKEND_URL}/signals/sample/{signal_type}")
        response.raise_for_status()
        return response.json()

def analyze_signal(payload: dict) -> dict:
    """Invia segnale al backend per analisi completa."""
    with httpx.Client(timeout=120.0) as client:
        response = client.post(f"{BACKEND_URL}/analysis/analyze", json=payload)
        response.raise_for_status()
        return response.json()

def check_backend_health() -> bool:
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(f"{BACKEND_URL}/health")
            return r.status_code == 200
    except Exception:
        return False

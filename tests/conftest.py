import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))


@pytest.fixture
def ecg_signal():
    import neurokit2 as nk
    sig = nk.ecg_simulate(duration=30, sampling_rate=500, noise=0.02)
    return np.array(sig), 500


@pytest.fixture
def ppg_signal():
    import neurokit2 as nk
    sig = nk.ppg_simulate(duration=30, sampling_rate=125, heart_rate=70)
    return np.array(sig), 125


@pytest.fixture
def short_ecg():
    import neurokit2 as nk
    sig = nk.ecg_simulate(duration=5, sampling_rate=500, noise=0.02)
    return np.array(sig), 500

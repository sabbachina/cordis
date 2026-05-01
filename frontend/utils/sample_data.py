import numpy as np
import neurokit2 as nk
import pandas as pd

def generate_sample_ecg(duration=30, fs=500, noise=0.02) -> tuple[np.ndarray, int]:
    signal = nk.ecg_simulate(duration=duration, sampling_rate=fs, noise=noise)
    return np.array(signal), fs

def generate_sample_ppg(duration=30, fs=125, heart_rate=70) -> tuple[np.ndarray, int]:
    signal = nk.ppg_simulate(duration=duration, sampling_rate=fs, heart_rate=heart_rate)
    return np.array(signal), fs

def signal_to_dataframe(signal: np.ndarray, fs: int) -> pd.DataFrame:
    time = np.arange(len(signal)) / fs
    return pd.DataFrame({"time_s": time, "amplitude": signal})

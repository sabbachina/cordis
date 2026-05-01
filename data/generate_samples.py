"""Genera segnali ECG e PPG sintetici di campione per testing."""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/ec2-user/SageMaker/AI_data/backend')


def generate_ecg_csv(output_path, duration=30, fs=500, noise=0.02):
    import neurokit2 as nk
    signal = nk.ecg_simulate(duration=duration, sampling_rate=fs, noise=noise)
    time = np.arange(len(signal)) / fs
    df = pd.DataFrame({"time_s": time, "ecg_mv": signal})
    df.to_csv(output_path, index=False)
    print(f"ECG saved: {output_path} ({len(signal)} samples @ {fs}Hz)")


def generate_ppg_csv(output_path, duration=30, fs=125, heart_rate=70):
    import neurokit2 as nk
    signal = nk.ppg_simulate(duration=duration, sampling_rate=fs, heart_rate=heart_rate)
    time = np.arange(len(signal)) / fs
    df = pd.DataFrame({"time_s": time, "ppg_au": signal})
    df.to_csv(output_path, index=False)
    print(f"PPG saved: {output_path} ({len(signal)} samples @ {fs}Hz)")


if __name__ == "__main__":
    generate_ecg_csv("/home/ec2-user/SageMaker/AI_data/data/samples/sample_ecg.csv")
    generate_ppg_csv("/home/ec2-user/SageMaker/AI_data/data/samples/sample_ppg.csv")
    print("Done!")

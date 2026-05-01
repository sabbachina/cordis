"""
Signal Loader — static factory methods to load physiological signals
from various file formats and data structures.

All methods return:
    tuple[np.ndarray, int]  →  (signal_array, sampling_rate_hz)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


class SignalLoader:
    """Collection of static loaders for ECG/PPG signals."""

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------
    @staticmethod
    def from_csv(
        filepath: str | Path,
        signal_col: str,
        time_col: Optional[str] = None,
        fs: Optional[int] = None,
    ) -> tuple[np.ndarray, int]:
        """
        Load a signal from a CSV file.

        Parameters
        ----------
        filepath   : path to the CSV file
        signal_col : column name containing amplitude values
        time_col   : optional column name with timestamps (seconds);
                     if provided and *fs* is None, fs is inferred from spacing
        fs         : explicit sampling rate in Hz; required when time_col is None

        Returns
        -------
        (signal, fs)
        """
        try:
            filepath = Path(filepath)
            if not filepath.is_file():
                raise ValueError(f"File not found: {filepath}")

            df = pd.read_csv(filepath)

            if signal_col not in df.columns:
                raise ValueError(
                    f"Column '{signal_col}' not found in CSV. "
                    f"Available columns: {list(df.columns)}"
                )

            signal = df[signal_col].dropna().to_numpy(dtype=np.float64)

            if signal.size < 10:
                raise ValueError(
                    f"Signal column '{signal_col}' contains fewer than 10 valid samples."
                )

            if time_col is not None:
                if time_col not in df.columns:
                    raise ValueError(
                        f"Time column '{time_col}' not found in CSV. "
                        f"Available columns: {list(df.columns)}"
                    )
                time = df[time_col].dropna().to_numpy(dtype=np.float64)
                if time.size >= 2:
                    dt = np.median(np.diff(time))
                    if dt <= 0:
                        raise ValueError("Time column must be strictly increasing.")
                    inferred_fs = int(round(1.0 / dt))
                    fs = inferred_fs if fs is None else fs
                else:
                    raise ValueError("Time column must have at least 2 valid rows.")

            if fs is None:
                raise ValueError(
                    "Sampling rate 'fs' must be provided when 'time_col' is not specified."
                )
            if fs <= 0:
                raise ValueError(f"Sampling rate must be positive; got {fs}.")

            return signal, int(fs)

        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Failed to load CSV '{filepath}': {exc}") from exc

    # ------------------------------------------------------------------
    # Excel
    # ------------------------------------------------------------------
    @staticmethod
    def from_excel(
        filepath: str | Path,
        signal_col: str,
        sheet_name: str | int = 0,
        fs: Optional[int] = None,
        time_col: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        """
        Load a signal from an Excel file (.xlsx / .xls).

        Parameters
        ----------
        filepath    : path to the Excel file
        signal_col  : column name containing amplitude values
        sheet_name  : sheet name or 0-based index (default: first sheet)
        fs          : explicit sampling rate in Hz
        time_col    : optional timestamp column for fs inference
        """
        try:
            filepath = Path(filepath)
            if not filepath.is_file():
                raise ValueError(f"File not found: {filepath}")

            df = pd.read_excel(filepath, sheet_name=sheet_name, engine="openpyxl")

            if signal_col not in df.columns:
                raise ValueError(
                    f"Column '{signal_col}' not found in sheet '{sheet_name}'. "
                    f"Available columns: {list(df.columns)}"
                )

            signal = df[signal_col].dropna().to_numpy(dtype=np.float64)

            if signal.size < 10:
                raise ValueError(
                    f"Signal column '{signal_col}' contains fewer than 10 valid samples."
                )

            if time_col is not None and time_col in df.columns:
                time = df[time_col].dropna().to_numpy(dtype=np.float64)
                if time.size >= 2:
                    dt = np.median(np.diff(time))
                    if dt > 0:
                        fs = fs or int(round(1.0 / dt))

            if fs is None:
                raise ValueError(
                    "Sampling rate 'fs' must be provided when 'time_col' is not specified."
                )
            if fs <= 0:
                raise ValueError(f"Sampling rate must be positive; got {fs}.")

            return signal, int(fs)

        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Failed to load Excel '{filepath}': {exc}") from exc

    # ------------------------------------------------------------------
    # EDF  (MNE)
    # ------------------------------------------------------------------
    @staticmethod
    def from_edf(
        filepath: str | Path,
        channel_name: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        """
        Load a single channel from an EDF/BDF file using MNE.

        Parameters
        ----------
        filepath     : path to .edf or .bdf file
        channel_name : exact channel label to extract;
                       if None, the first available channel is used
        """
        try:
            import mne  # type: ignore

            filepath = Path(filepath)
            if not filepath.is_file():
                raise ValueError(f"File not found: {filepath}")

            raw = mne.io.read_raw_edf(str(filepath), preload=True, verbose=False)
            available = raw.ch_names

            if channel_name is None:
                channel_name = available[0]
            elif channel_name not in available:
                # Case-insensitive fallback
                lower_map = {ch.lower(): ch for ch in available}
                match = lower_map.get(channel_name.lower())
                if match is None:
                    raise ValueError(
                        f"Channel '{channel_name}' not found in EDF. "
                        f"Available channels: {available}"
                    )
                channel_name = match

            idx = available.index(channel_name)
            data, _times = raw[idx, :]
            signal = data[0].astype(np.float64)

            fs = int(round(raw.info["sfreq"]))
            if fs <= 0:
                raise ValueError(f"Invalid sampling rate read from EDF: {fs}.")

            return signal, fs

        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Failed to load EDF '{filepath}': {exc}") from exc

    # ------------------------------------------------------------------
    # WFDB
    # ------------------------------------------------------------------
    @staticmethod
    def from_wfdb(
        record_path: str | Path,
        channel_idx: int = 0,
    ) -> tuple[np.ndarray, int]:
        """
        Load a channel from a WFDB record (PhysioNet format).

        Parameters
        ----------
        record_path : path to record *without* extension
                      (e.g. '/data/physionet/100')
        channel_idx : 0-based index of the channel to extract
        """
        try:
            import wfdb  # type: ignore

            record_path = str(record_path)
            record = wfdb.rdrecord(record_path)

            n_channels = record.p_signal.shape[1] if record.p_signal is not None else 0
            if n_channels == 0:
                raise ValueError("WFDB record contains no physical signals.")
            if channel_idx >= n_channels:
                raise ValueError(
                    f"channel_idx={channel_idx} out of range; "
                    f"record has {n_channels} channel(s)."
                )

            signal = record.p_signal[:, channel_idx].astype(np.float64)
            # Replace NaN (missing samples) with linear interpolation
            nan_mask = np.isnan(signal)
            if nan_mask.any():
                x = np.arange(len(signal))
                signal[nan_mask] = np.interp(x[nan_mask], x[~nan_mask], signal[~nan_mask])

            fs = int(record.fs)
            if fs <= 0:
                raise ValueError(f"Invalid sampling rate read from WFDB record: {fs}.")

            return signal, fs

        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Failed to load WFDB record '{record_path}': {exc}") from exc

    # ------------------------------------------------------------------
    # JSON dict
    # ------------------------------------------------------------------
    @staticmethod
    def from_json(data: dict) -> tuple[np.ndarray, int]:
        """
        Load a signal from a parsed JSON dictionary.

        Expected keys
        -------------
        'values'        : list of float amplitude samples  (required)
        'sampling_rate' : int Hz                           (required)

        Optional keys
        -------------
        'time' : list of float timestamps (unused for fs — 'sampling_rate' takes precedence)
        """
        try:
            if "values" not in data:
                raise ValueError(
                    "JSON dict must contain a 'values' key with the signal samples."
                )
            if "sampling_rate" not in data:
                raise ValueError(
                    "JSON dict must contain a 'sampling_rate' key with the Hz value."
                )

            raw_values = data["values"]
            if not isinstance(raw_values, (list, tuple, np.ndarray)):
                raise ValueError(
                    f"'values' must be a list of numbers; got {type(raw_values).__name__}."
                )

            signal = np.asarray(raw_values, dtype=np.float64)
            if signal.ndim != 1:
                raise ValueError(
                    f"'values' must be a 1-D array; got shape {signal.shape}."
                )
            if signal.size < 10:
                raise ValueError("'values' contains fewer than 10 samples.")

            fs = int(data["sampling_rate"])
            if fs <= 0:
                raise ValueError(
                    f"'sampling_rate' must be a positive integer; got {fs}."
                )

            return signal, fs

        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Failed to load signal from JSON dict: {exc}") from exc

    # ------------------------------------------------------------------
    # Raw array
    # ------------------------------------------------------------------
    @staticmethod
    def from_array(
        values: list[float] | np.ndarray,
        fs: int,
    ) -> tuple[np.ndarray, int]:
        """
        Wrap a Python list (or NumPy array) as a signal.

        Parameters
        ----------
        values : 1-D sequence of float amplitude samples
        fs     : sampling rate in Hz
        """
        try:
            signal = np.asarray(values, dtype=np.float64)

            if signal.ndim != 1:
                raise ValueError(
                    f"Input must be a 1-D sequence; got shape {signal.shape}."
                )
            if signal.size < 10:
                raise ValueError("Input contains fewer than 10 samples.")
            if not np.issubdtype(signal.dtype, np.floating) and not np.issubdtype(
                signal.dtype, np.integer
            ):
                raise ValueError("Input values must be numeric (int or float).")

            fs = int(fs)
            if fs <= 0:
                raise ValueError(f"Sampling rate must be positive; got {fs}.")

            return signal, fs

        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Failed to load signal from array: {exc}") from exc

"""
Test suite for SignalLoader — covers CSV, JSON, array, Excel and error paths.
"""
import os
import sys
import json
import tempfile

import numpy as np
import pandas as pd
import pytest

# sys.path already patched by conftest.py
from core.signal_loader import SignalLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ecg_csv(path: str, n: int = 200, fs: int = 500) -> None:
    """Write a minimal ECG CSV with time_s and ecg_mv columns."""
    time = np.arange(n) / fs
    values = np.sin(2 * np.pi * 1.0 * time)  # 1 Hz sine as dummy ECG
    df = pd.DataFrame({"time_s": time, "ecg_mv": values})
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Test 1 – from_csv: happy path
# ---------------------------------------------------------------------------

class TestFromCSV:
    def test_returns_tuple_array_int(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            tmp_path = f.name
        try:
            _make_ecg_csv(tmp_path)
            result = SignalLoader.from_csv(tmp_path, signal_col="ecg_mv",
                                           time_col="time_s")
            assert isinstance(result, tuple), "Result should be a tuple"
            signal, fs = result
            assert isinstance(signal, np.ndarray), "Signal should be np.ndarray"
            assert isinstance(fs, int), "Sampling rate should be int"
        finally:
            os.unlink(tmp_path)

    def test_signal_shape_and_length(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            tmp_path = f.name
        try:
            _make_ecg_csv(tmp_path, n=500, fs=500)
            signal, fs = SignalLoader.from_csv(tmp_path, signal_col="ecg_mv",
                                                time_col="time_s")
            assert signal.ndim == 1, "Signal must be 1-D"
            assert len(signal) == 500
            assert fs == 500
        finally:
            os.unlink(tmp_path)

    def test_fs_inferred_from_time_column(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            tmp_path = f.name
        try:
            _make_ecg_csv(tmp_path, n=200, fs=250)
            signal, fs = SignalLoader.from_csv(tmp_path, signal_col="ecg_mv",
                                                time_col="time_s")
            assert fs == 250
        finally:
            os.unlink(tmp_path)

    def test_explicit_fs_overrides_none(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            tmp_path = f.name
        try:
            _make_ecg_csv(tmp_path, n=200, fs=500)
            # Provide fs explicitly without time_col
            signal, fs = SignalLoader.from_csv(tmp_path, signal_col="ecg_mv",
                                                fs=200)
            assert fs == 200
        finally:
            os.unlink(tmp_path)

    def test_nonexistent_file_raises(self):
        """Missing file must raise ValueError or FileNotFoundError."""
        with pytest.raises((ValueError, FileNotFoundError)):
            SignalLoader.from_csv("/nonexistent/path/signal.csv",
                                  signal_col="ecg_mv", fs=500)

    def test_missing_signal_column_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            tmp_path = f.name
        try:
            _make_ecg_csv(tmp_path)
            with pytest.raises(ValueError, match="Column"):
                SignalLoader.from_csv(tmp_path, signal_col="wrong_col", fs=500)
        finally:
            os.unlink(tmp_path)

    def test_no_fs_and_no_time_col_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            tmp_path = f.name
        try:
            _make_ecg_csv(tmp_path)
            with pytest.raises(ValueError):
                SignalLoader.from_csv(tmp_path, signal_col="ecg_mv")
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Test 2 – from_json
# ---------------------------------------------------------------------------

class TestFromJSON:
    def test_valid_dict(self):
        data = {"values": list(np.random.randn(300)), "sampling_rate": 500}
        signal, fs = SignalLoader.from_json(data)
        assert isinstance(signal, np.ndarray)
        assert signal.dtype == np.float64
        assert fs == 500
        assert len(signal) == 300

    def test_missing_values_key_raises(self):
        with pytest.raises(ValueError, match="values"):
            SignalLoader.from_json({"sampling_rate": 500})

    def test_missing_sampling_rate_raises(self):
        with pytest.raises(ValueError, match="sampling_rate"):
            SignalLoader.from_json({"values": [1.0] * 20})

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError):
            SignalLoader.from_json({"values": [1.0] * 5, "sampling_rate": 500})

    def test_returns_1d_array(self):
        data = {"values": list(range(100)), "sampling_rate": 100}
        signal, fs = SignalLoader.from_json(data)
        assert signal.ndim == 1
        assert fs == 100

    def test_negative_sampling_rate_raises(self):
        with pytest.raises(ValueError):
            SignalLoader.from_json({"values": [1.0] * 50, "sampling_rate": -1})


# ---------------------------------------------------------------------------
# Test 3 – from_array
# ---------------------------------------------------------------------------

class TestFromArray:
    def test_list_of_floats(self):
        values = [float(i) for i in range(100)]
        signal, fs = SignalLoader.from_array(values, fs=100)
        assert isinstance(signal, np.ndarray)
        assert signal.dtype == np.float64
        assert signal.shape == (100,)
        assert fs == 100

    def test_numpy_array_input(self):
        values = np.random.randn(250)
        signal, fs = SignalLoader.from_array(values, fs=500)
        assert signal.shape == (250,)
        assert isinstance(fs, int)

    def test_shape_is_1d(self):
        values = list(range(50))
        signal, _ = SignalLoader.from_array(values, fs=50)
        assert signal.ndim == 1

    def test_dtype_is_float64(self):
        values = list(range(50))
        signal, _ = SignalLoader.from_array(values, fs=50)
        assert signal.dtype == np.float64

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            SignalLoader.from_array([1.0, 2.0, 3.0], fs=100)

    def test_zero_fs_raises(self):
        with pytest.raises(ValueError):
            SignalLoader.from_array(list(range(50)), fs=0)

    def test_negative_fs_raises(self):
        with pytest.raises(ValueError):
            SignalLoader.from_array(list(range(50)), fs=-10)

    def test_2d_input_raises(self):
        arr = np.ones((10, 10))
        with pytest.raises(ValueError):
            SignalLoader.from_array(arr, fs=100)


# ---------------------------------------------------------------------------
# Test 4 – from_excel (requires openpyxl)
# ---------------------------------------------------------------------------

class TestFromExcel:
    def _make_excel(self, path: str, n: int = 200, fs: int = 250) -> None:
        time = np.arange(n) / fs
        values = np.sin(2 * np.pi * 1.0 * time)
        df = pd.DataFrame({"time_s": time, "ecg_mv": values})
        df.to_excel(path, index=False, engine="openpyxl")

    def test_basic_load(self):
        pytest.importorskip("openpyxl")
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            tmp_path = f.name
        try:
            self._make_excel(tmp_path, n=200, fs=250)
            signal, fs = SignalLoader.from_excel(tmp_path, signal_col="ecg_mv",
                                                  time_col="time_s")
            assert isinstance(signal, np.ndarray)
            assert len(signal) == 200
            assert fs == 250
        finally:
            os.unlink(tmp_path)

    def test_explicit_fs_excel(self):
        pytest.importorskip("openpyxl")
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            tmp_path = f.name
        try:
            self._make_excel(tmp_path, n=150, fs=125)
            signal, fs = SignalLoader.from_excel(tmp_path, signal_col="ecg_mv",
                                                  fs=125)
            assert fs == 125
            assert len(signal) == 150
        finally:
            os.unlink(tmp_path)

    def test_nonexistent_excel_raises(self):
        with pytest.raises(ValueError):
            SignalLoader.from_excel("/nonexistent/file.xlsx",
                                    signal_col="ecg_mv", fs=500)

    def test_missing_column_excel_raises(self):
        pytest.importorskip("openpyxl")
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            tmp_path = f.name
        try:
            self._make_excel(tmp_path)
            with pytest.raises(ValueError, match="Column"):
                SignalLoader.from_excel(tmp_path, signal_col="nonexistent",
                                        fs=500)
        finally:
            os.unlink(tmp_path)

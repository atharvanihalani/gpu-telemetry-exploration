"""Window chunking and feature extraction from GPU telemetry CSVs."""

import pandas as pd
import numpy as np
from dataclasses import dataclass


DEFAULT_WINDOW_S = 60
SM_ACTIVE_FLOOR = 0.01  # ignore timesteps with near-zero SM for tensor ratio

# Autocorrelation: search for periodicity between 0.2s and 5s
AUTOCORR_MIN_LAG_S = 0.2
AUTOCORR_MAX_LAG_S = 5.0


@dataclass
class WindowFeatures:
    window_idx: int
    start_s: float
    end_s: float
    mean_power: float
    tensor_sm_ratio: float
    power_std_temporal: float
    nvlink_autocorr_peak: float
    n_samples: int


def _gpu_autocorr_peak(signal: np.ndarray, samples_per_sec: float) -> float:
    """Peak autocorrelation of a single GPU's NVLink signal in the lag range."""
    signal = signal - signal.mean()
    if len(signal) < 10 or np.std(signal) < 1e-6:
        return 0.0

    autocorr = np.correlate(signal, signal, mode="full")
    autocorr = autocorr / autocorr[len(autocorr) // 2]  # normalize lag-0 to 1.0
    mid = len(autocorr) // 2

    min_lag = max(2, int(AUTOCORR_MIN_LAG_S * samples_per_sec))
    max_lag = int(AUTOCORR_MAX_LAG_S * samples_per_sec)
    max_lag = min(max_lag, len(autocorr) - mid - 1)

    if max_lag <= min_lag:
        return 0.0

    return float(np.max(autocorr[mid + min_lag : mid + max_lag + 1]))


def _nvlink_autocorr_peak(chunk: pd.DataFrame) -> float:
    """Max NVLink TX autocorrelation peak across all GPUs in a window."""
    duration = chunk["elapsed"].max() - chunk["elapsed"].min()
    if duration <= 0:
        return 0.0
    samples_per_sec = len(chunk) / len(chunk["gpu"].unique()) / duration

    peaks = []
    for _, gpu_data in chunk.groupby("gpu"):
        signal = gpu_data["nvlink_tx_bytes_s"].values
        peaks.append(_gpu_autocorr_peak(signal, samples_per_sec))

    return max(peaks) if peaks else 0.0


def load_telemetry(csv_path: str, phase: str = "steady") -> pd.DataFrame:
    """Load a telemetry CSV and filter to the given phase.

    If the requested phase doesn't exist, returns all data with a warning.
    """
    df = pd.read_csv(csv_path)
    if phase and phase in df["phase"].values:
        df = df[df["phase"] == phase]
    return df


def extract_features(
    df: pd.DataFrame, window_s: float = DEFAULT_WINDOW_S
) -> list[WindowFeatures]:
    """Chop telemetry into time windows and compute features per window."""
    t0 = df["timestamp"].min()
    df = df.copy()
    df["elapsed"] = df["timestamp"] - t0

    duration = df["elapsed"].max()
    n_windows = max(1, int(np.floor(duration / window_s)))

    windows = []
    for i in range(n_windows):
        start = i * window_s
        end = (i + 1) * window_s
        chunk = df[(df["elapsed"] >= start) & (df["elapsed"] < end)]
        if chunk.empty:
            continue

        # Mean power across all GPUs and timesteps
        mean_power = chunk["power_w"].mean()

        # Tensor/SM ratio: per-sample ratio where SM is active, then mean
        active = chunk[chunk["sm_active"] > SM_ACTIVE_FLOOR]
        if len(active) > 0:
            ratios = active["tensor_active"] / active["sm_active"]
            tensor_sm_ratio = ratios.mean()
        else:
            tensor_sm_ratio = 0.0

        # Power temporal std: per-GPU std over time, averaged across GPUs
        per_gpu_std = chunk.groupby("gpu")["power_w"].std().fillna(0)
        power_std_temporal = per_gpu_std.mean()

        # NVLink autocorrelation: max peak across GPUs
        nvlink_autocorr_peak = _nvlink_autocorr_peak(chunk)

        windows.append(WindowFeatures(
            window_idx=i,
            start_s=start,
            end_s=end,
            mean_power=mean_power,
            tensor_sm_ratio=tensor_sm_ratio,
            power_std_temporal=power_std_temporal,
            nvlink_autocorr_peak=nvlink_autocorr_peak,
            n_samples=len(chunk),
        ))

    return windows

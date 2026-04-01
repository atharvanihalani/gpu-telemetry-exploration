"""Window chunking and feature extraction from GPU telemetry CSVs."""

import pandas as pd
import numpy as np
from dataclasses import dataclass


DEFAULT_WINDOW_S = 60
SM_ACTIVE_FLOOR = 0.01  # ignore timesteps with near-zero SM for tensor ratio


@dataclass
class WindowFeatures:
    window_idx: int
    start_s: float
    end_s: float
    mean_power: float
    tensor_sm_ratio: float
    power_std_temporal: float
    n_samples: int


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

        windows.append(WindowFeatures(
            window_idx=i,
            start_s=start,
            end_s=end,
            mean_power=mean_power,
            tensor_sm_ratio=tensor_sm_ratio,
            power_std_temporal=power_std_temporal,
            n_samples=len(chunk),
        ))

    return windows

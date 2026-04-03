"""Threshold-based classification rules for GPU telemetry windows."""

from dataclasses import dataclass, field
from .features import WindowFeatures


# Default thresholds — physically motivated from H100 telemetry
POWER_THRESHOLD = 400.0       # watts — sustained high power (catches T1, T2, T3, T5)
TENSOR_RATIO_THRESHOLD = 0.25  # tensor_active / sm_active (catches T6, E3 on-periods)
NVLINK_AUTOCORR_THRESHOLD = 0.3  # periodicity in NVLink TX (catches allreduce heartbeat)


@dataclass
class WindowVerdict:
    window_idx: int
    start_s: float
    end_s: float
    is_training: bool
    triggered_rules: list[str]
    mean_power: float
    tensor_sm_ratio: float
    nvlink_autocorr_peak: float


@dataclass
class Thresholds:
    power: float = POWER_THRESHOLD
    tensor_ratio: float = TENSOR_RATIO_THRESHOLD
    nvlink_autocorr: float = NVLINK_AUTOCORR_THRESHOLD


def classify_window(
    feat: WindowFeatures, thresholds: Thresholds = Thresholds()
) -> WindowVerdict:
    """Apply threshold rules to a single window's features."""
    triggered = []

    if feat.mean_power > thresholds.power:
        triggered.append("power")
    if feat.tensor_sm_ratio > thresholds.tensor_ratio:
        triggered.append("tensor_ratio")
    if feat.nvlink_autocorr_peak > thresholds.nvlink_autocorr:
        triggered.append("nvlink_autocorr")

    return WindowVerdict(
        window_idx=feat.window_idx,
        start_s=feat.start_s,
        end_s=feat.end_s,
        is_training=len(triggered) > 0,
        triggered_rules=triggered,
        mean_power=feat.mean_power,
        tensor_sm_ratio=feat.tensor_sm_ratio,
        nvlink_autocorr_peak=feat.nvlink_autocorr_peak,
    )


def classify_windows(
    features: list[WindowFeatures], thresholds: Thresholds = Thresholds()
) -> list[WindowVerdict]:
    """Classify all windows in a telemetry trace."""
    return [classify_window(f, thresholds) for f in features]


def overall_verdict(verdicts: list[WindowVerdict]) -> tuple[bool, int, int]:
    """Summarize: (any_training_detected, n_flagged, n_total)."""
    n_flagged = sum(1 for v in verdicts if v.is_training)
    return n_flagged > 0, n_flagged, len(verdicts)

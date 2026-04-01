"""
Shared background telemetry collector — DCGM backend at 10Hz.

Polls DCGM for standard + profiling metrics from a daemon thread.
The main workload script calls start() / set_phase() / stop() — the
collector handles everything else.

Requires nv-hostengine running (DCGM daemon).

Usage:
    collector = TelemetryCollector("data/t1_telemetry.csv")
    collector.start()
    collector.set_phase("warmup")
    # ... workload ...
    collector.set_phase("steady")
    # ... workload ...
    collector.stop()
"""

import csv
import sys
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# DCGM Python bindings (shipped with datacenter-gpu-manager, not on PyPI)
# ---------------------------------------------------------------------------
_DCGM_BINDINGS_PATH = "/usr/local/dcgm/bindings/python3/"
if _DCGM_BINDINGS_PATH not in sys.path:
    sys.path.insert(0, _DCGM_BINDINGS_PATH)

try:
    import dcgm_structs
    import dcgm_agent
    import dcgm_fields
    import dcgmvalue
except ImportError as exc:
    raise RuntimeError(
        f"DCGM Python bindings not found at {_DCGM_BINDINGS_PATH}. "
        "Install datacenter-gpu-manager and ensure nv-hostengine is running."
    ) from exc

# ---------------------------------------------------------------------------
# CSV columns (20 total)
# ---------------------------------------------------------------------------
FIELDS = [
    "timestamp",
    "phase",
    "gpu",
    "gpu_model",
    "power_w",
    "temp_c",
    "mem_used_mib",
    "mem_total_mib",
    "gpu_util_pct",
    "mem_util_pct",
    "sm_active",
    "tensor_active",
    "dram_active",
    "fp16_active",
    "pcie_tx_bytes_s",
    "pcie_rx_bytes_s",
    "nvlink_tx_bytes_s",
    "nvlink_rx_bytes_s",
    "throttle_reasons",
    "energy_mj",
]

# ---------------------------------------------------------------------------
# DCGM field IDs to watch (15 per-sample fields)
# ---------------------------------------------------------------------------
_DCGM_FIELD_IDS = [
    155,   # power_w           (double, watts)
    150,   # temp_c            (int64, celsius)
    252,   # mem_used_mib      (int64, MiB)
    203,   # gpu_util_pct      (int64, %)
    204,   # mem_util_pct      (int64, %)
    1002,  # sm_active         (double, ratio 0-1)
    1004,  # tensor_active     (double, ratio 0-1)
    1005,  # dram_active       (double, ratio 0-1)
    1008,  # fp16_active       (double, ratio 0-1)
    1009,  # pcie_tx_bytes_s   (double, bytes/sec)
    1010,  # pcie_rx_bytes_s   (double, bytes/sec)
    1011,  # nvlink_tx_bytes_s (double, bytes/sec)
    1012,  # nvlink_rx_bytes_s (double, bytes/sec)
    112,   # throttle_reasons  (int64, bitmask)
    156,   # energy_mj         (int64, cumulative millijoules)
]

# Map DCGM field ID -> CSV column name
_FIELD_TO_COLUMN = {
    155:  "power_w",
    150:  "temp_c",
    252:  "mem_used_mib",
    203:  "gpu_util_pct",
    204:  "mem_util_pct",
    1002: "sm_active",
    1004: "tensor_active",
    1005: "dram_active",
    1008: "fp16_active",
    1009: "pcie_tx_bytes_s",
    1010: "pcie_rx_bytes_s",
    1011: "nvlink_tx_bytes_s",
    1012: "nvlink_rx_bytes_s",
    112:  "throttle_reasons",
    156:  "energy_mj",
}


class TelemetryCollector:
    def __init__(self, output_path: str, interval_s: float = 0.1):
        self.output_path = Path(output_path)
        self.interval_s = interval_s
        self._phase = "init"
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._lock = threading.Lock()

        # DCGM handles (set in start / _connect_dcgm)
        self._dcgm_handle = None
        self._gpu_group = None
        self._field_group = None
        self._gpu_ids = []
        self._gpu_names = []
        self._mem_totals = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        self._connect_dcgm()
        self._setup_watches()

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        self._writer.writeheader()
        self._file.flush()

        self._thread.start()
        print(f"[telemetry] collecting → {self.output_path}  (DCGM, {self.interval_s}s)")

    def set_phase(self, phase: str):
        with self._lock:
            self._phase = phase
        print(f"[telemetry] phase = {phase}")

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        self._file.close()
        self._cleanup_dcgm()
        print(f"[telemetry] done → {self.output_path}")

    # ------------------------------------------------------------------
    # DCGM setup
    # ------------------------------------------------------------------

    def _connect_dcgm(self):
        """Initialise DCGM library, connect to nv-hostengine, enumerate GPUs."""
        try:
            dcgm_structs._dcgmInit()
        except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_ALREADY_INITIALIZED):
            pass  # library already loaded in this process

        try:
            dcgm_structs.dcgmLib.dcgmInit()
        except dcgm_structs.dcgmExceptionClass(dcgm_structs.DCGM_ST_ALREADY_INITIALIZED):
            pass

        try:
            self._dcgm_handle = dcgm_agent.dcgmConnect("127.0.0.1")
        except Exception as exc:
            raise RuntimeError(
                "Cannot connect to nv-hostengine at 127.0.0.1:5555. "
                "Start it with: nv-hostengine"
            ) from exc

        # Enumerate GPUs
        gpu_ids = dcgm_agent.dcgmGetAllSupportedDevices(self._dcgm_handle)
        self._gpu_ids = list(gpu_ids)
        n = len(self._gpu_ids)

        # Read static per-GPU info (gpu_model via field 50, mem_total via field 252)
        self._gpu_names = []
        self._mem_totals = []
        for gid in self._gpu_ids:
            try:
                attrs = dcgm_agent.dcgmGetDeviceAttributes(self._dcgm_handle, gid)
                name = attrs.identifiers.deviceName
                mem_total = attrs.memoryUsage.fbTotal
            except Exception:
                name = "unknown"
                mem_total = 0
            self._gpu_names.append(name)
            self._mem_totals.append(mem_total)

        print(f"[telemetry] DCGM connected — {n}x {self._gpu_names[0]}")

    def _setup_watches(self):
        """Create GPU group, field group, and start watching at the configured rate."""
        handle = self._dcgm_handle

        # GPU group containing all GPUs (unique name to avoid collisions across processes)
        import os
        suffix = os.getpid()
        self._gpu_group = dcgm_agent.dcgmGroupCreate(
            handle, dcgm_structs.DCGM_GROUP_EMPTY, f"telemetry_gpus_{suffix}"
        )
        for gid in self._gpu_ids:
            dcgm_agent.dcgmGroupAddDevice(handle, self._gpu_group, gid)

        # Field group with all per-sample fields
        self._field_group = dcgm_agent.dcgmFieldGroupCreate(
            handle, _DCGM_FIELD_IDS, f"telemetry_fields_{suffix}"
        )

        # Watch: updateFreq in microseconds, maxKeepAge in seconds, maxKeepSamples
        update_freq_us = int(self.interval_s * 1_000_000)
        dcgm_agent.dcgmWatchFields(
            handle,
            self._gpu_group,
            self._field_group,
            update_freq_us,
            5.0,   # maxKeepAge (seconds)
            50,    # maxKeepSamples
        )

        # Let profiling counters warm up
        time.sleep(0.5)

    # ------------------------------------------------------------------
    # Polling loop
    # ------------------------------------------------------------------

    def _run(self):
        while not self._stop_event.is_set():
            loop_start = time.time()
            ts = loop_start

            with self._lock:
                phase = self._phase

            # Force a field update so we get fresh values
            try:
                dcgm_agent.dcgmUpdateAllFields(self._dcgm_handle, 1)
            except Exception:
                pass  # best-effort; reads below may return slightly stale data

            for idx, gid in enumerate(self._gpu_ids):
                row = self._read_gpu(idx, gid, ts, phase)
                self._writer.writerow(row)

            self._file.flush()

            elapsed = time.time() - loop_start
            remaining = self.interval_s - elapsed
            if remaining > 0:
                self._stop_event.wait(remaining)

    def _read_gpu(self, idx: int, gid: int, ts: float, phase: str) -> dict:
        """Read all watched fields for one GPU, return a CSV row dict."""
        row = {
            "timestamp": f"{ts:.3f}",
            "phase": phase,
            "gpu": idx,
            "gpu_model": self._gpu_names[idx],
            "mem_total_mib": self._mem_totals[idx],
        }

        try:
            values = dcgm_agent.dcgmEntityGetLatestValues(
                self._dcgm_handle,
                dcgm_fields.DCGM_FE_GPU,
                gid,
                _DCGM_FIELD_IDS,
            )
        except Exception:
            # DCGM read failed — fill all metric columns with blanks
            for col in _FIELD_TO_COLUMN.values():
                row[col] = ""
            return row

        for v in values:
            col = _FIELD_TO_COLUMN.get(v.fieldId)
            if col is None:
                continue

            # Check for blank / not-available sentinels
            if v.fieldType == ord('d'):
                if dcgmvalue.DCGM_FP64_IS_BLANK(v.value.dbl):
                    row[col] = ""
                else:
                    # Format doubles: power to 1dp, ratios to 6dp, bytes to 0dp
                    val = v.value.dbl
                    if col == "power_w":
                        row[col] = f"{val:.1f}"
                    elif col in ("sm_active", "tensor_active", "dram_active", "fp16_active"):
                        row[col] = f"{val:.6f}"
                    elif col in ("pcie_tx_bytes_s", "pcie_rx_bytes_s",
                                 "nvlink_tx_bytes_s", "nvlink_rx_bytes_s"):
                        row[col] = f"{val:.0f}"
                    else:
                        row[col] = val
            elif v.fieldType == ord('i'):
                if dcgmvalue.DCGM_INT64_IS_BLANK(v.value.i64):
                    row[col] = ""
                else:
                    row[col] = v.value.i64
            else:
                row[col] = ""

        return row

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup_dcgm(self):
        """Unwatch fields, destroy groups, disconnect."""
        handle = self._dcgm_handle
        if handle is None:
            return

        try:
            if self._field_group is not None and self._gpu_group is not None:
                dcgm_agent.dcgmUnwatchFields(handle, self._gpu_group, self._field_group)
        except Exception:
            pass

        try:
            if self._field_group is not None:
                dcgm_agent.dcgmFieldGroupDestroy(handle, self._field_group)
        except Exception:
            pass

        try:
            if self._gpu_group is not None:
                dcgm_agent.dcgmGroupDestroy(handle, self._gpu_group)
        except Exception:
            pass

        try:
            dcgm_agent.dcgmDisconnect(handle)
        except Exception:
            pass

        self._dcgm_handle = None
        self._gpu_group = None
        self._field_group = None

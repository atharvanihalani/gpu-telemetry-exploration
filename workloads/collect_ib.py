"""
InfiniBand telemetry collector — sysfs counters at 10Hz.

Polls cumulative byte/packet counters from /sys/class/infiniband/mlx5_ibN/
and computes per-second rates. Runs independently of the DCGM collector;
outputs to its own CSV, synced by timestamp.

Counters are maintained by ConnectX firmware, exposed through the kernel
RDMA subsystem — completely independent of the GPU driver stack.

Usage:
    collector = IBCollector("data/t10_node0_ib.csv")
    collector.start()
    collector.set_phase("steady")
    collector.stop()
"""

import csv
import os
import threading
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Discover IB devices
# ---------------------------------------------------------------------------
IB_SYSFS_BASE = "/sys/class/infiniband"

# Counters to read per port
_COUNTERS = ["port_xmit_data", "port_rcv_data", "port_xmit_packets", "port_rcv_packets"]


def _discover_ib_devices() -> list[str]:
    """Find all mlx5_ibN devices with active ports."""
    devices = []
    if not os.path.isdir(IB_SYSFS_BASE):
        return devices
    for name in sorted(os.listdir(IB_SYSFS_BASE)):
        if not name.startswith("mlx5_ib"):
            continue
        state_path = os.path.join(IB_SYSFS_BASE, name, "ports", "1", "state")
        try:
            with open(state_path) as f:
                if "ACTIVE" in f.read():
                    devices.append(name)
        except OSError:
            continue
    return devices


def _read_counter(device: str, counter: str) -> int:
    """Read a single sysfs counter value."""
    path = os.path.join(IB_SYSFS_BASE, device, "ports", "1", "counters", counter)
    with open(path) as f:
        return int(f.read().strip())


# ---------------------------------------------------------------------------
# CSV columns
# ---------------------------------------------------------------------------
def _build_fields(devices: list[str]) -> list[str]:
    """Build CSV column names: timestamp, phase, then per-device rate columns."""
    fields = ["timestamp", "phase"]
    for dev in devices:
        short = dev.replace("mlx5_", "")  # ib0, ib1, ...
        fields.append(f"{short}_tx_bytes_s")
        fields.append(f"{short}_rx_bytes_s")
        fields.append(f"{short}_tx_pkts_s")
        fields.append(f"{short}_rx_pkts_s")
    # Aggregate columns
    fields.append("total_tx_bytes_s")
    fields.append("total_rx_bytes_s")
    return fields


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------
class IBCollector:
    def __init__(self, output_path: str, interval_s: float = 0.1):
        self.output_path = Path(output_path)
        self.interval_s = interval_s
        self._phase = "init"
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._lock = threading.Lock()

        self._devices = _discover_ib_devices()
        self._fields = _build_fields(self._devices)
        self._prev_counters: dict | None = None
        self._prev_time: float | None = None

    def start(self):
        if not self._devices:
            print("[ib] no active IB devices found — skipping")
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self._fields)
        self._writer.writeheader()
        self._file.flush()

        self._thread.start()
        print(f"[ib] collecting → {self.output_path}  ({len(self._devices)} ports, {self.interval_s}s)")

    def set_phase(self, phase: str):
        with self._lock:
            self._phase = phase

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join()
        if hasattr(self, "_file"):
            self._file.close()
        print(f"[ib] done → {self.output_path}")

    def _read_all_counters(self) -> dict[str, int]:
        """Read all counters for all devices."""
        counters = {}
        for dev in self._devices:
            for counter in _COUNTERS:
                key = f"{dev}_{counter}"
                try:
                    counters[key] = _read_counter(dev, counter)
                except OSError:
                    counters[key] = 0
        return counters

    def _run(self):
        # Prime the counters (first read is just to establish baseline)
        self._prev_counters = self._read_all_counters()
        self._prev_time = time.time()
        self._stop_event.wait(self.interval_s)

        while not self._stop_event.is_set():
            loop_start = time.time()
            ts = loop_start

            with self._lock:
                phase = self._phase

            current = self._read_all_counters()
            dt = ts - self._prev_time

            if dt > 0:
                row = {
                    "timestamp": f"{ts:.3f}",
                    "phase": phase,
                }

                total_tx = 0
                total_rx = 0

                for dev in self._devices:
                    short = dev.replace("mlx5_", "")

                    tx_bytes = (current[f"{dev}_port_xmit_data"] - self._prev_counters[f"{dev}_port_xmit_data"]) / dt
                    rx_bytes = (current[f"{dev}_port_rcv_data"] - self._prev_counters[f"{dev}_port_rcv_data"]) / dt
                    tx_pkts = (current[f"{dev}_port_xmit_packets"] - self._prev_counters[f"{dev}_port_xmit_packets"]) / dt
                    rx_pkts = (current[f"{dev}_port_rcv_packets"] - self._prev_counters[f"{dev}_port_rcv_packets"]) / dt

                    # IB counters are in 4-byte words for data, raw count for packets
                    row[f"{short}_tx_bytes_s"] = f"{tx_bytes * 4:.0f}"
                    row[f"{short}_rx_bytes_s"] = f"{rx_bytes * 4:.0f}"
                    row[f"{short}_tx_pkts_s"] = f"{tx_pkts:.0f}"
                    row[f"{short}_rx_pkts_s"] = f"{rx_pkts:.0f}"

                    total_tx += tx_bytes * 4
                    total_rx += rx_bytes * 4

                row["total_tx_bytes_s"] = f"{total_tx:.0f}"
                row["total_rx_bytes_s"] = f"{total_rx:.0f}"

                self._writer.writerow(row)
                self._file.flush()

            self._prev_counters = current
            self._prev_time = ts

            elapsed = time.time() - loop_start
            remaining = self.interval_s - elapsed
            if remaining > 0:
                self._stop_event.wait(remaining)

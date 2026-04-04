"""
BMC telemetry collector — IPMI sensors at ~1Hz.

Reads system power and per-GPU temperatures from the baseboard management
controller via ipmitool. These measurements are hardware-level and
independent of the GPU driver — useful for consistency checks against
DCGM readings.

BMC polling is slow (~0.5-1s per ipmitool call), so this runs at ~1Hz.

Usage:
    collector = BMCCollector("data/t10_node0_bmc.csv")
    collector.start()
    collector.set_phase("steady")
    collector.stop()
"""

import csv
import os
import re
import subprocess
import threading
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# IPMI sensor reading
# ---------------------------------------------------------------------------

# Sensors to read: (sensor_name, csv_column, unit_expected)
_SENSORS = [
    ("SYS_POWER", "sys_power_w", "Watts"),
    ("GPU0_PROC", "gpu0_bmc_temp_c", "degrees C"),
    ("GPU1_PROC", "gpu1_bmc_temp_c", "degrees C"),
    ("GPU2_PROC", "gpu2_bmc_temp_c", "degrees C"),
    ("GPU3_PROC", "gpu3_bmc_temp_c", "degrees C"),
    ("GPU4_PROC", "gpu4_bmc_temp_c", "degrees C"),
    ("GPU5_PROC", "gpu5_bmc_temp_c", "degrees C"),
    ("GPU6_PROC", "gpu6_bmc_temp_c", "degrees C"),
    ("GPU7_PROC", "gpu7_bmc_temp_c", "degrees C"),
]

FIELDS = ["timestamp", "phase"] + [s[1] for s in _SENSORS]


def _check_ipmitool() -> bool:
    """Check if ipmitool is available and can read sensors."""
    try:
        result = subprocess.run(
            ["sudo", "ipmitool", "sdr", "get", "SYS_POWER"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _read_sensors() -> dict[str, str]:
    """Read all configured sensors in a single ipmitool call."""
    sensor_names = [s[0] for s in _SENSORS]

    try:
        result = subprocess.run(
            ["sudo", "ipmitool", "sdr", "get"] + sensor_names,
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}

    if result.returncode != 0:
        return {}

    # Parse output: "Sensor Reading        : 2100 (+/- 0) Watts"
    values = {}
    current_sensor = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("Sensor ID"):
            # Extract sensor name: "Sensor ID              : SYS_POWER (0xe9)"
            match = re.search(r":\s+(\S+)", line)
            if match:
                current_sensor = match.group(1)
        elif line.startswith("Sensor Reading") and current_sensor:
            # Extract numeric value: "Sensor Reading        : 2100 (+/- 0) Watts"
            match = re.search(r":\s+([\d.]+)", line)
            if match:
                # Find the corresponding CSV column
                for sensor_name, csv_col, _ in _SENSORS:
                    if sensor_name == current_sensor:
                        values[csv_col] = match.group(1)
                        break

    return values


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------
class BMCCollector:
    def __init__(self, output_path: str, interval_s: float = 2.0):
        self.output_path = Path(output_path)
        self.interval_s = interval_s
        self._phase = "init"
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._lock = threading.Lock()

    def start(self):
        if not _check_ipmitool():
            print("[bmc] ipmitool not available or no BMC access — skipping")
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        self._writer.writeheader()
        self._file.flush()

        self._thread.start()
        print(f"[bmc] collecting → {self.output_path}  ({self.interval_s}s)")

    def set_phase(self, phase: str):
        with self._lock:
            self._phase = phase

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join()
        if hasattr(self, "_file"):
            self._file.close()
        print(f"[bmc] done → {self.output_path}")

    def _run(self):
        while not self._stop_event.is_set():
            loop_start = time.time()

            with self._lock:
                phase = self._phase

            values = _read_sensors()

            row = {
                "timestamp": f"{loop_start:.3f}",
                "phase": phase,
            }
            for _, csv_col, _ in _SENSORS:
                row[csv_col] = values.get(csv_col, "")

            self._writer.writerow(row)
            self._file.flush()

            elapsed = time.time() - loop_start
            remaining = self.interval_s - elapsed
            if remaining > 0:
                self._stop_event.wait(remaining)

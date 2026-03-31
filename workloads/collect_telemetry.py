"""
Shared background telemetry collector.

Polls pynvml every second from a daemon thread. The main workload script
calls start() / set_phase() / stop() — the collector handles everything else.

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
import threading
import time
from pathlib import Path

import pynvml


FIELDS = [
    "timestamp",
    "phase",
    "gpu",
    "power_w",
    "sm_util_pct",
    "mem_util_pct",
    "mem_used_mib",
    "mem_total_mib",
    "temp_c",
    "pcie_tx_kib",
    "pcie_rx_kib",
    "clock_sm_mhz",
    "clock_mem_mhz",
]


class TelemetryCollector:
    def __init__(self, output_path: str, interval_s: float = 1.0):
        self.output_path = Path(output_path)
        self.interval_s = interval_s
        self._phase = "init"
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._lock = threading.Lock()

    def start(self):
        pynvml.nvmlInit()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        self._writer.writeheader()
        self._file.flush()
        self._thread.start()
        print(f"[telemetry] collecting → {self.output_path}")

    def set_phase(self, phase: str):
        with self._lock:
            self._phase = phase
        print(f"[telemetry] phase = {phase}")

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        self._file.close()
        pynvml.nvmlShutdown()
        print(f"[telemetry] done → {self.output_path}")

    def _run(self):
        n_gpus = pynvml.nvmlDeviceGetCount()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n_gpus)]

        while not self._stop_event.is_set():
            ts = time.time()
            with self._lock:
                phase = self._phase

            for i, h in enumerate(handles):
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                except pynvml.NVMLError:
                    power = float("nan")

                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    sm_util = util.gpu
                    mem_util = util.memory
                except pynvml.NVMLError:
                    sm_util = mem_util = float("nan")

                try:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    mem_used = mem.used / (1024 ** 2)
                    mem_total = mem.total / (1024 ** 2)
                except pynvml.NVMLError:
                    mem_used = mem_total = float("nan")

                try:
                    temp = pynvml.nvmlDeviceGetTemperature(
                        h, pynvml.NVML_TEMPERATURE_GPU
                    )
                except pynvml.NVMLError:
                    temp = float("nan")

                try:
                    pcie_tx = pynvml.nvmlDeviceGetPcieThroughput(
                        h, pynvml.NVML_PCIE_UTIL_TX_BYTES
                    )
                    pcie_rx = pynvml.nvmlDeviceGetPcieThroughput(
                        h, pynvml.NVML_PCIE_UTIL_RX_BYTES
                    )
                except pynvml.NVMLError:
                    pcie_tx = pcie_rx = float("nan")

                try:
                    clk_sm = pynvml.nvmlDeviceGetClockInfo(
                        h, pynvml.NVML_CLOCK_SM
                    )
                    clk_mem = pynvml.nvmlDeviceGetClockInfo(
                        h, pynvml.NVML_CLOCK_MEM
                    )
                except pynvml.NVMLError:
                    clk_sm = clk_mem = float("nan")

                self._writer.writerow({
                    "timestamp": f"{ts:.3f}",
                    "phase": phase,
                    "gpu": i,
                    "power_w": f"{power:.1f}",
                    "sm_util_pct": sm_util,
                    "mem_util_pct": mem_util,
                    "mem_used_mib": f"{mem_used:.0f}",
                    "mem_total_mib": f"{mem_total:.0f}",
                    "temp_c": temp,
                    "pcie_tx_kib": pcie_tx,
                    "pcie_rx_kib": pcie_rx,
                    "clock_sm_mhz": clk_sm,
                    "clock_mem_mhz": clk_mem,
                })

            self._file.flush()
            # sleep for the remainder of the interval
            elapsed = time.time() - ts
            remaining = self.interval_s - elapsed
            if remaining > 0:
                time.sleep(remaining)

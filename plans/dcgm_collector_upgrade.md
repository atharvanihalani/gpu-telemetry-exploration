# Plan: Replace pynvml Collector with Unified DCGM at 10Hz

## Context

The current `workloads/collect_telemetry.py` uses pynvml at 1Hz and captures 14 columns. It's missing critical signals for training detection — notably NVLink tx/rx bytes, tensor core utilization, and DRAM activity — which are only available via DCGM profiling fields. The 1Hz rate also risks aliasing the NVLink allreduce heartbeat (sub-second bursts).

We'll replace the pynvml backend with DCGM, add the missing metrics, and increase polling to 10Hz. Same public API, no workload script changes needed.

## Verified on this node

- DCGM 3.3.9, nv-hostengine running, 8 H100s visible
- Python bindings at `/usr/local/dcgm/bindings/python3/`
- Init sequence: `dcgm_structs._dcgmInit()` → `dcgm_structs.dcgmLib.dcgmInit()` → `dcgm_agent.dcgmConnect('127.0.0.1')`
- Standard fields (155, 150, 252, etc.) and profiling Group A fields (1002-1012) **coexist in the same watch group** — confirmed
- 17 fields × 8 GPUs reads in ~16ms — well within 100ms budget
- Blank sentinels: `dcgmvalue.DCGM_FP64_IS_BLANK` / `DCGM_INT64_IS_BLANK` available

## New CSV columns (20 total)

| CSV Column | DCGM Field | Type | Notes |
|---|---|---|---|
| `timestamp` | — | float | Unix epoch, ms precision |
| `phase` | — | string | Set by workload script |
| `gpu` | — | int | GPU index 0-7 |
| `gpu_model` | 50 | string | Read once at init |
| `power_w` | 155 | double | Watts |
| `temp_c` | 150 | int | Celsius |
| `mem_used_mib` | 252 | int | Framebuffer used |
| `mem_total_mib` | — | int | Read once from field 252 at init (constant) |
| `gpu_util_pct` | 203 | int | Legacy SM util (backward compat) |
| `mem_util_pct` | 204 | int | Memory controller util |
| `sm_active` | 1002 | double | Ratio 0-1, granular SM activity |
| `tensor_active` | 1004 | double | Ratio 0-1, tensor core usage — key training signal |
| `dram_active` | 1005 | double | Ratio 0-1, HBM bandwidth |
| `fp16_active` | 1008 | double | Ratio 0-1, FP16/BF16 pipe |
| `pcie_tx_bytes_s` | 1009 | double | Bytes/sec |
| `pcie_rx_bytes_s` | 1010 | double | Bytes/sec |
| `nvlink_tx_bytes_s` | 1011 | double | Bytes/sec — allreduce heartbeat |
| `nvlink_rx_bytes_s` | 1012 | double | Bytes/sec |
| `throttle_reasons` | 112 | int | Bitmask — relevant for E1 power cap |
| `energy_mj` | 156 | int | Cumulative millijoules (diff for rate) |

Dropped from old format: `clock_sm_mhz`, `clock_mem_mhz`, `pcie_tx_kib`, `pcie_rx_kib` (replaced by profiling-based pcie_tx/rx_bytes_s which are more accurate).

## Implementation — single file rewrite

**File:** `workloads/collect_telemetry.py`

### Structure

```
# Module-level: DCGM imports (sys.path insert + imports, guarded by try/except)
# Module-level: FIELDS list, DCGM_FIELD_IDS list, field-ID-to-column mapping

class TelemetryCollector:
    __init__(output_path, interval_s=0.1)   # default changed from 1.0 to 0.1
    start()                                  # DCGM connect + group/field setup + thread start
    set_phase(phase)                         # same as current (lock-protected)
    stop()                                   # thread join + DCGM cleanup
    _connect_dcgm()                          # init sequence + GPU enumeration
    _setup_watches()                         # create GPU group + field group + watch
    _run()                                   # daemon thread polling loop
    _read_gpu(gpu_id, ts, phase) -> dict     # read fields for one GPU, handle blanks
    _cleanup_dcgm()                          # unwatch + destroy groups + disconnect
```

### Key details

1. **DCGM init** in `start()` via `_connect_dcgm()`: load library, init, connect to 127.0.0.1. Read gpu_model (field 50) and mem_total once per GPU. Fail fast with clear error if nv-hostengine not running.

2. **Watch setup** via `_setup_watches()`: create GPU group with all devices, create field group with the 15 per-sample fields, call `dcgmWatchFields(updateFreq=100000, maxKeepAge=5.0, maxKeepSamples=50)`. Sleep 500ms for profiling counters to warm up.

3. **Polling loop** in `_run()`: same structure as current — loop until stop event, read all GPUs, write rows, flush, sleep remainder. Use `self._stop_event.wait(remaining)` instead of `time.sleep()` for faster shutdown. Wrap per-GPU reads in try/except — write blank row on DCGM error rather than crashing.

4. **Blank handling**: check `dcgmvalue.DCGM_FP64_IS_BLANK(v)` / `DCGM_INT64_IS_BLANK(v)` before writing values. Blanks written as empty string (pandas reads as NaN).

5. **Cleanup** in `stop()`: unwatch fields, destroy field group, destroy GPU group, disconnect, close CSV.

6. **No pynvml fallback**: require DCGM. Raise `RuntimeError` with actionable message if unavailable.

## What doesn't change

- Public API: `TelemetryCollector(path, interval_s)`, `.start()`, `.set_phase()`, `.stop()`
- Import: `from workloads.collect_telemetry import TelemetryCollector`
- None of the 15 workload scripts need modification
- CSV format: DictWriter, one row per GPU per sample
- Phase lifecycle: init → loading → warmup → steady → cooldown

## CSV size impact

Old (1Hz): ~2,400 rows / 196KB per 5-min run
New (10Hz): ~24,000 rows / ~2MB per 5-min run
15 conditions total: ~30MB. Negligible.

## Verification

1. Run collector standalone for 5s, inspect CSV output — all 20 columns populated, no blanks on idle node
2. Run `torchrun --nproc_per_node=8 workloads/train_t1.py` — verify nvlink_tx/rx_bytes_s shows allreduce bursts, tensor_active > 0, power near TDP
3. Run `python workloads/infer_i2.py` — verify nvlink near zero, tensor_active near zero, power low
4. Spot-check that existing workload scripts import and run without changes

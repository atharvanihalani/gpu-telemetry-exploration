---
name: Project context and current state
description: What the GPU telemetry project is building toward and where it currently stands
type: project
---

Goal: build a technical foundation for verifying that AI labs have paused pre-training runs during a coordinated slowdown. Core question: can you distinguish training from inference (or idle) from GPU telemetry signals?

**Current state (as of 2026-04-01):**
- Session 1: passive exploration — pynvml/DCGM baseline, NVLink topology, idle metrics
- Session 2: active fingerprinting — ran T1 (DDP training) and I2 (streaming inference), collected labeled 5-min telemetry CSVs, built comparison notebook
- Session 3: implemented all remaining workload scripts (T2–T6, I3, I4, E1–E5, B1). Moved to H100 node. Made code hardware-agnostic.
- Session 4: upgraded collector from pynvml 1Hz to DCGM 10Hz (20 columns). Recollected T1. Ran and collected T2–T6, E3, E4, I2 (recollected), I3, B1. 4 workloads shelved.

**Collector upgrade (session 4):** `collect_telemetry.py` now uses DCGM at 10Hz instead of pynvml at 1Hz. Output CSVs have 20 columns including DCGM profiling fields (sm_active, sm_occupancy, tensor_active, fp16_active, dram_active, nvlink_tx_bytes, nvlink_rx_bytes, pcie_tx_bytes, pcie_rx_bytes).

**Collected workloads (11 of 15):** T1, T2, T3, T4, T5, T6, E3, E4, I2, I3, B1
- All collected on H100 SXM5-80GB at DCGM 10Hz
- Data lives in `data/*_telemetry.csv`

**Shelved workloads (4):**
- E1 (power-capped training) — RunPod H100 pods block `nvmlDeviceSetPowerManagementLimit`; can't set power limits as a cloud tenant
- E2 (cover traffic orchestrator) — subprocess coordination issues; deferred
- E5 (smoothed allreduce) — NCCL timeout during collection; deferred
- I4 (speculative decoding) — transformers compatibility issues; deferred

**Key finding from session 4:** `fp16_active` is always 0 across all workloads — models use bf16, which shows up in `tensor_active` instead. Don't use fp16_active as a signal.

**Key finding from session 2**: zero signal overlap between T1 and I2 at 1Hz. The allreduce heartbeat (periodic synchronized power dips across all 8 GPUs) is clearly visible in training and completely absent in inference.

**Next priority**: build a multi-condition comparison analysis notebook covering all 11 collected workloads. Answer: what signal survives each evasion, what combination is hardest to simultaneously fake.

**How to apply:** data collection phase is largely complete. Focus shifts to analysis — building the comparison notebook and deriving detection-relevant conclusions from the full telemetry dataset.

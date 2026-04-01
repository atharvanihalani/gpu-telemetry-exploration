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
- Session 5: built multi-condition comparison notebook locally (macOS). 12 plots covering distributions, timeseries, heatmaps, synchrony, tensor ratio, NVLink symmetry, correlations.

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

**Key findings from session 5 (comparison analysis):**
- Each evasion breaks ONE detection signal but leaves others intact: E3 breaks sustained power, E4 breaks NVLink. Neither breaks everything simultaneously → multi-signal detection is the path forward.
- E3 (intermittent training) on-periods look identical to T1 — a detector checking any 30s window catches it. The on/off periodicity is itself detectable.
- E4 (PCIe-only allreduce) kills NVLink heartbeat but power stays elevated (~350W vs ~100W inference), SM/tensor still high. Power+SM alone would catch it.
- Tensor core ratio (tensor_active / sm_active) is a strong standalone classifier — training clusters at 0.5–0.9 with high SM, inference clusters in the low-SM corner.
- NVLink TX vs RX is highly symmetric for training (allreduce) and near-zero for inference. Clean separation.
- Signal correlations differ structurally between training (tight power-SM-tensor-NVLink coupling) and inference (weaker, different structure).

**Comparison notebook design decisions (session 5):**
- Binary color scheme: dark/light red = training+evasion ("flag"), dark/light blue = inference+baseline ("benign")
- Two-section layout: training+evasion (2×4 grid top) vs inference+baseline (1×3 bottom)
- All data truncated to 300s steady-state for consistency
- 12 plots saved to `plots/`, old plots archived to `plots/archived/`

**Session 6 — classifier built (2026-04-01):**
- Built `classifier/` package: 3 threshold rules (power >400W, tensor/SM ratio >0.25, power temporal std >40W) on 60s windows
- 11/11 conditions classified correctly, 0 false positives, 0 false negatives
- E4 (PCIe-only allreduce) is the hardest case — only caught by power_std rule
- Dropped cross-GPU synchrony metric as unprincipled (relies on implementation artifact)
- Deleted partial data files (i4, e2, e5) — need rerun on RunPod
- E5 to be rerun with smaller model (T2's 136M) to avoid NCCL timeout
- Notebook: `notebooks/classify.ipynb` — thin UI over `classifier/` modules

**Next priority**: run threshold sweep to check margins. Rerun E2/E5/I4 on RunPod. Consider what E2 (cover traffic) does to the classifier. Write up findings for verification proposal.

**How to apply:** classifier v1 is working. Project is transitioning from exploration to building a detection system. Future work should stress-test the classifier against harder evasion scenarios (especially E2 cover traffic).

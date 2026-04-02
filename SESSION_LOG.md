# Session Log — GPU Telemetry Exploration

Detailed per-session narrative. Referenced from CLAUDE.md but not loaded every conversation turn.

---

## Session 1 — Baseline Exploration (2026-03-30, A100 RunPod)

Initial hands-on session. Built `telemetry_exploration.ipynb` — enumerates GPUs, collects 60s idle baseline at 1Hz via pynvml, visualizes NVLink topology, plots 6-panel dashboard + heatmaps. Set up DCGM and collected per-link NVLink data. Shelved stress tests for later.

---

## Session 2 — Mock Workload Fingerprinting (2026-03-30/31, A100 RunPod)

Shifted to active fingerprinting. Built T1 (DDP pre-training, 3.37B params) and I2 (streaming Llama-3.1-8B inference). Collected labeled telemetry. Key finding: zero overlap between T1 and I2 on every signal. The GPU×time power heatmap shows synchronized allreduce heartbeat in T1 (periodic power dips across all GPUs) — completely absent in I2. Created `mock_conditions.md` taxonomy of all planned conditions.

**Files:** `workloads/train_t1.py`, `workloads/infer_i2.py`, `workloads/collect_telemetry.py`, `data/t1_telemetry.csv`, `data/i2_telemetry.csv`, `notebooks/comparison.ipynb`

---

## Session 3 — Full Workload Implementation (2026-03-31, H100 RunPod)

Implemented all remaining workload scripts. Moved to H100. Made code hardware-agnostic.

**New scripts:** T2, T3, T4, T5, T6, I3, I4, E1, E2, E3, E4, E5, B1 — see `workloads/` directory.

**Design patterns:** thin wrappers (E4, E5 import from T1), clean forks (T2, T3, T5, E3), orchestrators (E2), external libs (I3/vLLM, I4/HF assisted generation), safety guards (E1 power limit restore).

**Skipped:** I1 (batched forward — bracketed by I2+I3), B2 (checkpoint I/O), B3 (CUDA graph warmup).

---

## Session 4 — Full Telemetry Collection (2026-04-01, H100 RunPod)

Rewrote `collect_telemetry.py` from pynvml to DCGM at 10Hz. 20 columns (added sm_active, tensor_active, dram_active, fp16_active, nvlink_tx/rx_bytes_s, pcie_tx/rx_bytes_s, throttle_reasons, energy_mj). Collected 11 conditions on H100.

| Condition | Rows | Notes |
|---|---|---|
| T1 | 26K | Large DDP pre-training (3.37B) |
| T2 | 25K | Small pre-training (136M) |
| T3 | 27K | Gradient accumulation (16 steps) |
| T4 | 64K | Pipeline parallelism (PiPPy, 8 stages) |
| T5 | 30K | Gradient checkpointing (6.71B) |
| T6 | 27K | FSDP + CPU offload (ZeRO-3) |
| E3 | 55K | Intermittent training (30s on / 10s off) |
| E4 | 51K | PCIe-only allreduce (NVLink disabled) |
| I2 | 38K | Streaming autoregressive inference |
| I3 | 40K | High-throughput vLLM (8-GPU TP) |
| B1 | 26K | Idle with Llama-3.1-8B loaded |

**Shelved:** E1 (RunPod blocks power limits), E2 (HF token propagation), E5 (NCCL timeout), I4 (meta tensor bug).

**Bug fixes:** T4 bf16 dtype mismatch, T5 OOM probe, T1 TELEMETRY_DISABLED check, DCGM PID-suffixed group names, I3 deprecated vLLM param, I4 draft model swap.

**Key observation:** `fp16_active` always 0.000 — models use bf16, activity shows in `tensor_active` instead.

---

## Session 5 — Multi-Condition Comparison Analysis (2026-04-01, local macOS)

Pure analysis session. Built full comparison notebook analyzing all 11 datasets. 12 plots in `plots/`.

**Design:** Binary color scheme (red=training/evasion, blue=inference/baseline). Two-section layout. 300s truncation. Shared scales.

**Key findings:**
1. Every evasion breaks ONE signal but leaves others intact
2. Multi-signal detection is robust
3. Tensor core ratio is a strong standalone classifier
4. NVLink TX/RX symmetry separates allreduce from everything else
5. Cross-GPU synchrony shows heartbeat even in E4 (via PCIe)
6. Signal correlation structure differs between training and inference

---

## Session 6 — Threshold-Based Classifier (2026-04-01, local macOS)

Built simple 3-rule threshold classifier in `classifier/`. No ML, no fitting.

| Rule | Threshold | Physical basis |
|---|---|---|
| `mean_power` | > 400W | Sustained high compute |
| `tensor_sm_ratio` | > 0.25 | Matmul-dominated workload |
| `power_std_temporal` | > 40W | Step-cycle oscillation |

**Validation:** 11/11 correct. OR logic — evasion must defeat ALL rules. Per-window (60s) classification.

Deleted partial E2/E5/I4 data from session 4.

---

## Session 7 — E2/E5 Collection + NVLink Autocorrelation (2026-04-02, H100 RunPod)

Collected E2 and E5 (13 conditions total). Added 4th classifier rule: NVLink autocorrelation.

### E2 — Cover Traffic (first successful run)
- GPUs 0-3: inference, GPUs 4-7: training (4-GPU DDP)
- "HF token propagation" bug was a non-issue — token just wasn't in env
- Aggregate tensor_sm_ratio 0.242 — just below 0.25 threshold. Cover traffic nearly dodges it.

### E5 — Smoothed Allreduce (third attempt)
- 128MB/1ch → NCCL timeout. 16MB/4ch → no visible effect on H100. **32MB/2ch → worked.**
- Power 625W (vs T1's 685W), heartbeat autocorrelation still strong (0.739)

### 4th Rule — NVLink Autocorrelation
- `nvlink_autocorr_peak > 0.3`: max autocorrelation of NVLink TX in 0.2-5s lag range, across GPUs
- **12/13 correct. I3 false positive** — tensor-parallel inference has periodic NVLink from per-layer activation exchanges (autocorr 0.699). Fix: compound rule (autocorr AND power), deferred.

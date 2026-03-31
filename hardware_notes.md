# Hardware Notes — A100 vs H100

We switch between A100 and H100 RunPod nodes. The workload scripts and telemetry collector are hardware-agnostic (the `gpu_model` CSV column records which GPU produced each row). This doc covers what actually differs between the two.

---

## Spec comparison

| | A100 SXM4-80GB | H100 SXM5-80GB |
|---|---|---|
| Architecture | Ampere (GA100) | Hopper (GH100) |
| TDP | ~400W | ~700W |
| Memory | 80 GB HBM2e | 80 GB HBM3 |
| Memory BW | ~2.0 TB/s | ~3.35 TB/s |
| FP16/BF16 TFLOPS | ~312 | ~989 |
| FP8 TFLOPS | N/A | ~1979 |
| NVLink version | 3.0 | 4.0 |
| NVLink ports | 12 (L0–L11) | 18 (L0–L17) |
| NVLink BW (bidirectional) | 600 GB/s | 900 GB/s |
| Max SM clock | ~1410 MHz | ~1980 MHz |
| Memory clock | ~1593 MHz | ~2619 MHz |
| NVSwitch | 4 chips, all-to-all | 4 chips, all-to-all |
| Transformer Engine / FP8 | No | Yes |

---

## What changes for our workloads

### T1 (DDP pre-training)
- **Model config stays the same** — OOM boundary is memory-limited (80GB on both), not compute-limited. The 3.37B config works on both; the 6.4B config OOMs on both.
- **Training power will be higher** — expect ~600–700W on H100 vs ~400W on A100.
- **Steps/sec will be ~2–3× faster** on H100 (more SMs, higher clocks, higher memory BW). The allreduce heartbeat period will be shorter.
- **NVLink traffic patterns are the same** (all-reduce every step), just faster and across 18 links instead of 12.

### I2 (streaming inference)
- **Works identically** — Llama-3.1-8B fits easily on both.
- **Inference power will be somewhat higher** on H100 (higher idle floor, higher clocks when active).
- **Tokens/sec will be higher** (memory BW is 1.7× higher).

### Evasion experiments
- **E1 (power cap)**: Target ~20–25% of TDP, not a fixed wattage. On A100: ~100–150W. On H100: ~150–175W. The goal is to land in the inference power range for that GPU.
- **E4 (PCIe-only allreduce)**: Same env vars (`NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1`), works on both.
- **E5 (smoothed allreduce)**: Same NCCL tuning env vars, works on both.

### DCGM
- NVLink bandwidth fields: `409–420` on A100 (12 links), `409–426` on H100 (18 links).
- All other field IDs (power, SM util, memory, temp) are the same.
- DCGM may need to be installed fresh on each pod (`apt-get install datacenter-gpu-manager`).

---

## Checklist when switching nodes

1. **No code changes needed** — scripts auto-detect GPU count and use pynvml generically.
2. **Telemetry CSVs are tagged** — `gpu_model` column identifies the hardware.
3. **Don't compare absolute values across GPUs** — compare *patterns* (allreduce heartbeat, cross-GPU synchronization, temporal variability). Ratios between training and inference are more meaningful than raw watts.
4. **Re-run baseline if needed** — idle power, clocks, and thermal behavior differ.
5. **DCGM install** — likely not pre-installed; run the setup commands from CLAUDE.md.
6. **E1 power cap** — recalculate target wattage based on `nvidia-smi --query-gpu=power.default_limit --format=csv`.

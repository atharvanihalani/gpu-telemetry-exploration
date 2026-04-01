# GPU Telemetry Exploration — Handoff for Claude Code

## What this node is

RunPod 8×A100 SXM4-80GB pod. Root access. Ubuntu 22.04, CUDA 12.4, PyTorch 2.4.0 pre-installed.

This is a **temporary exploratory node** — session cost ~$12/hr, stop it when done. No persistent work needs to be saved here except the collected CSV data and plots (copy those off before stopping).

---

## What we're doing and why

Atharva is building a trusted third-party org to verify that frontier AI labs have actually paused pre-training runs during a coordinated AI slowdown. The core technical problem: how do you tell, from outside, whether a GPU cluster is secretly training a model vs sitting idle or running inference?

**This session's goal**: Get hands-on with raw GPU telemetry. Dump everything the node exposes. Understand what signals actually look like at the hardware level before building any verification proposals.

This is **open-ended exploration**, not hypothesis testing. The deliverable is understanding, not a polished artifact.

---

## The main notebook

`telemetry_exploration.ipynb` — run it top to bottom.

**What it does:**
1. Installs `nvidia-ml-py3`, pandas, matplotlib, seaborn
2. Enumerates all 8 GPUs via pynvml, shows driver/CUDA versions
3. Discovers NVLink topology (which GPUs are wired together, how many active ports)
4. Visualizes the NVLink connectivity matrix
5. Collects 60s of idle baseline metrics at 1s intervals: power, SM util, memory, temp, PCIe, clocks
6. Saves to `telemetry_baseline.csv`
7. Plots a 6-panel time-series dashboard + GPU×time heatmaps
8. Sets up DCGM and runs `dcgmi dmon` to collect per-link NVLink data (fields 409–420, L0–L11)
9. Plots NVLink per-link heatmap (GPU × NVLink port) and rate timeline
10. Has a [SHELVED] section at the bottom with stress test workloads — uncomment to run

**Expected runtime**: ~5 minutes for the baseline collection, ~2 more minutes for DCGM collection.

---

## Hardware specs

We flip between A100 and H100 nodes. Both are 8-GPU, 80GB, all-to-all NVSwitch. The workload scripts are hardware-agnostic — the `gpu_model` column in telemetry CSVs records which GPU produced each row.

| Item | A100 SXM4 | H100 SXM5 |
|---|---|---|
| GPU | NVIDIA A100-SXM4-80GB | NVIDIA H100 80GB HBM3 |
| GPU count | 8 | 8 |
| NVLink version | NVLink 3.0 | NVLink 4.0 |
| NVLink ports per GPU | 12 (L0–L11) | 18 (L0–L17) |
| NVLink bandwidth | 600 GB/s bidirectional | 900 GB/s bidirectional |
| Memory | 80 GB HBM2e | 80 GB HBM3 |
| GPU TDP | ~400W | ~700W |
| Max SM clock | ~1410 MHz | ~1980 MHz |
| NVSwitch | 4 chips (all-to-all) | 4 chips (all-to-all) |
| Scale-out | InfiniBand | InfiniBand |

**On an idle node, expect roughly:**
- Power: ~60–80W per GPU (both A100 and H100)
- SM utilization: ~0%
- NVLink traffic: ~0 (maybe a few MB of control-plane)
- Memory: whatever is loaded (should be near-zero if nothing's running)

**What changes between A100 and H100** (see also `hardware_notes.md`):
- Training power envelope: ~400W (A100) vs ~700W (H100)
- NVLink DCGM field range: 409–420 (A100, 12 links) vs 409–426 (H100, 18 links)
- Steps/sec will differ (~2–3× faster on H100 for same model)
- E1 power cap target must be set relative to TDP, not a fixed wattage

---

## Why these signals matter

The project's core insight: **training and inference look very different at the hardware level**, and these differences are hard to fake at scale.

| Signal | Training signature | Inference signature | Why it matters |
|---|---|---|---|
| **Power** | Flat, sustained, high for hours/days | Spiky, demand-following | Hardest to hide; measurable from facility power meters |
| **NVLink traffic** | Massive, periodic, symmetric bursts (all-reduce) | Sparse, asymmetric | The smoking gun — gradient sync has a unique "heartbeat" pattern |
| **SM utilization** | Relentlessly high | Bursty | Training is always compute-bound |
| **Memory used** | Very high (weights + gradients + optimizer states) | Lower (weights + KV cache only) | Optimizer states (Adam = 8× model params) don't exist in inference |
| **PCIe** | Periodic (data loads, checkpoint writes) | Constant (token I/O) | Pattern differs |

**The most important crux**: temporal pattern over time, not instantaneous snapshot. A single high-utilization moment tells you nothing. Sustained flat power + periodic NVLink bursts over hours = training.

**NVLink all-reduce** is the single strongest training signal: every backward pass, every GPU shares gradients with every other GPU simultaneously. On an 8-GPU node, this means 8 GPUs all spiking their NVLink traffic at the same moment, symmetrically. Nothing in inference looks like that.

---

## DCGM setup (if not already running)

DCGM (Data Center GPU Manager) is NVIDIA's telemetry daemon. It exposes per-link NVLink counters that `nvidia-smi` doesn't surface cleanly.

```bash
# Check if installed
which dcgmi

# Install if not present (Ubuntu 22.04)
apt-get update -q && apt-get install -y datacenter-gpu-manager

# Start the daemon
nv-hostengine

# Verify it can see GPUs
dcgmi discovery -l

# Find available NVLink field IDs (409-420 on A100, 409-426 on H100)
dcgmi fieldids -l | grep -i nvlink
```

**Key DCGM field IDs (common across A100/H100):**
- `203` = SM utilization (%)
- `155` = Power draw (W)
- `150` = Temperature (°C)
- `100` / `101` = Framebuffer free / used (MiB)
- `409–420` = NVLink bandwidth L0–L11 (cumulative bytes per link — diff consecutive readings for rate)
- `421–426` = NVLink bandwidth L12–L17 (**H100 only** — 18 links vs A100's 12)

**Note**: Fields 409+ are **cumulative counters**, not rates. To get GB/s, diff two readings and divide by the interval.

**Known DCGM gotcha**: Some versions have a bug where Fabric Manager state isn't accurately reported on out-of-band queries, and there's a race condition in the FSP NVLink query that can crash the FSP (fixed in newer versions). If `dcgmi dmon` hangs or produces garbage, try restarting `nv-hostengine`.

---

## Tooling available

`pynvml` (nvidia-ml-py3) — the primary Python interface. Wraps NVML, same as `nvidia-smi`. Gives: power, SM util, memory, temp, clocks, PCIe throughput, NVLink link state.

`nvidia-smi` — CLI, always available. Useful for quick checks:
```bash
nvidia-smi                          # overview
nvidia-smi -l 1                     # refresh every 1s
nvidia-smi dmon -s u -d 1          # utilization stream
nvidia-smi dmon -s n -d 1          # NVLink aggregate stream
nvidia-smi topo -m                  # NVLink topology matrix
```

`dcgmi` — DCGM CLI, requires `nv-hostengine` running. More powerful than nvidia-smi for per-link NVLink data.

**What's NOT accessible on this pod** (cloud tenant limitation):
- BMC / Redfish (provider's management network — would give independent power readings)
- BlueField DPU state (provider's SDN)
- TOR switch traffic (provider's network infrastructure)

For *signal identification* this is fine — DCGM reads the same underlying hardware counters. The trust/tamper-resistance question (can you fake these readings?) is a separate problem, deferred to later.

---

## After running the notebook

Things worth exploring beyond the notebook:

**1. Run `nvidia-smi topo -m`** to see the full NVLink topology matrix directly. Compare with what the notebook's connectivity matrix shows.

**2. Uncomment and run the stress tests** in the notebook's shelved section:
- Compute stress (matmul loop): should spike power to near TDP (~400W A100, ~700W H100), SM util to ~100%, NVLink stays near zero
- All-reduce stress (`torchrun --nproc_per_node=8 allreduce_stress.py`): should show the NVLink heartbeat pattern

**3. Compare baseline vs stress plots side by side** — this is the core of what we're trying to understand.

**4. Check DCGM profiling fields** if available:
```bash
dcgmi fieldids -l | grep -i prof
# Look for: DCGM_FI_PROF_PIPE_TENSOR_ACTIVE, DCGM_FI_PROF_SM_ACTIVE, DCGM_FI_PROF_DRAM_ACTIVE
```
These give more granular compute pipeline metrics than the standard utilization fields.

---

## Session 2 — Mock Workload Fingerprinting (2026-03-30/31)

### What we built

The focus shifted from passive observation to active fingerprinting: running controlled mock workloads and collecting labeled telemetry to establish ground-truth signatures for training vs. inference.

**New files:**
```
mock_conditions.md          — full taxonomy of workload conditions to test (training variants,
                              inference variants, adversarial/evasion, other baselines)
plans/t1_i2_implemented.md  — detailed design decisions for T1 + I2 scripts
workloads/
  collect_telemetry.py      — shared background telemetry collector (pynvml, 1Hz, CSV output)
  train_t1.py               — T1: DDP pre-training mock (torchrun --nproc_per_node=8)
  infer_i2.py               — I2: streaming autoregressive inference (Llama-3.1-8B)
data/
  t1_telemetry.csv          — 270s steady-state training telemetry
  i2_telemetry.csv          — 368s steady-state inference telemetry
notebooks/
  comparison.ipynb          — side-by-side T1 vs I2 analysis (4 plot types)
comparison_timeseries.png
comparison_distributions.png
comparison_heatmaps.png
comparison_power_variability.png
```

### T1 — Large Pre-Training

- Architecture: GPT-style decoder-only transformer, **3.37B params** (d_model=3072, 28 layers, 24 heads)
- Note: 6.4B config OOMed — activations at batch=4, seq=2048 consumed ~75GB leaving no room for AdamW states
- Parallelism: DDP, 8 GPUs, `torchrun --nproc_per_node=8 workloads/train_t1.py`
- Data: synthetic random token tensors (telemetry-equivalent to real data)
- Precision: bf16 weights/grads, fp32 AdamW states
- Duration: 5 min (30s warmup, 270s steady, 5s cooldown)

### I2 — Streaming Autoregressive Inference

- Model: **Llama-3.1-8B** (public/ungated on HuggingFace, ~16GB download)
- Setup: one model per GPU, 8 independent streams, no tensor parallelism
- Generation: greedy decode, 64-token prompt → 512 new tokens per request, repeated
- Duration: 5 min (loading + 30s warmup + 368s steady + 5s cooldown)
- Note: Llama-3.1-8B is ungated — no HF token needed

### Key findings (collected on A100 SXM4 — absolute values will differ on H100)

Steady-state signal comparison (mean across 8 GPUs):

| Signal | T1 training | I2 inference | Ratio |
|---|---|---|---|
| Power | 392W | 88W | **4.4×** |
| SM utilization | 100% | 4.8% | **21×** |
| Memory used | 66,858 MiB | 17,451 MiB | **3.8×** |
| Temperature | 66°C | 34°C | **1.9×** |
| Power cross-GPU std | 31.3W | 2.0W | **16×** |

**Most important finding**: the GPU×time power heatmap shows vertical white stripes in T1 — all 8 GPUs simultaneously dropping power at regular intervals (~15–20s period = one training step). This is the allreduce sync heartbeat. It's completely absent in I2. A detector looking for periodic, synchronized power dips across all GPUs would be hard to evade.

**Zero overlap** between T1 and I2 on every signal at 1Hz polling. Any single signal is sufficient to classify correctly under these conditions. The interesting question is what happens under evasion conditions — see `mock_conditions.md`.

### Installed packages (pip, system Python)

```
nvidia-ml-py    — pynvml
pandas
matplotlib
seaborn
transformers
accelerate
huggingface_hub
vllm==0.6.3.post1  — added in session 3 for I3 (pinned to PyTorch 2.4.x compat)
```

### Running the workloads

```bash
# Training variants
torchrun --nproc_per_node=8 workloads/train_t1.py   # T1 — large DDP pre-training (3.37B)
torchrun --nproc_per_node=8 workloads/train_t2.py   # T2 — small DDP pre-training (136M)
torchrun --nproc_per_node=8 workloads/train_t3.py   # T3 — gradient accumulation (16 steps)
torchrun --nproc_per_node=8 workloads/train_t4.py   # T4 — pipeline parallelism (PiPPy, 8 stages)
torchrun --nproc_per_node=8 workloads/train_t5.py   # T5 — gradient checkpointing (6.71B, fallback 3.37B)
torchrun --nproc_per_node=8 workloads/train_t6.py   # T6 — FSDP + CPU offload (ZeRO-3)

# Inference variants
python workloads/infer_i2.py                         # I2 — streaming autoregressive (Llama-3.1-8B)
python workloads/infer_i3.py                         # I3 — high-throughput vLLM (8-GPU TP)
python workloads/infer_i4.py                         # I4 — speculative decoding (draft + verifier)

# Evasion conditions
torchrun --nproc_per_node=8 workloads/train_e1.py   # E1 — power-capped training (~22% TDP)
python workloads/run_e2.py                           # E2 — cover traffic (train 4-7 + infer 0-3)
torchrun --nproc_per_node=8 workloads/train_e3.py   # E3 — intermittent training (30s on / 10s off)
torchrun --nproc_per_node=8 workloads/train_e4.py   # E4 — PCIe-only allreduce (NVLink disabled)
torchrun --nproc_per_node=8 workloads/train_e5.py   # E5 — smoothed allreduce (Ring, 128MB, 1ch)

# Baselines
python workloads/baseline_b1.py                      # B1 — idle with model loaded
```

### What's been collected vs pending

**Collected (session 2, on A100):** T1, I2 → `data/t1_telemetry.csv`, `data/i2_telemetry.csv`

**Implemented but not yet run:** T2, T3, T4, T5, T6, I3, I4, E1, E2, E3, E4, E5, B1

**Skipped (deprioritized):** I1 (batched forward pass — bracketed by I2+I3), B2 (checkpoint I/O), B3 (CUDA graph warmup)

### Priority order for next collection runs

1. **E1** — power-capped training (tests most-cited detection signal)
2. **E4** — PCIe-only allreduce (tests strongest detection signal — NVLink heartbeat)
3. **I3** — high-throughput inference (hardest inference case, closest to training)
4. **T3** — gradient accumulation (stresses temporal pattern detector)
5. **E3** — intermittent training (breaks sustained-power assumption)
6. Everything else as time/budget allows

---

## Session 3 — Full Workload Implementation (2026-03-31)

### What we built

Implemented all remaining workload scripts from `mock_conditions.md`. Moved to an H100 SXM5-80GB node. Made all code hardware-agnostic (A100/H100).

**New workload scripts:**
```
workloads/
  train_t2.py       — T2: small pre-training (136M params, edge case)
  train_t3.py       — T3: gradient accumulation (ACCUM_STEPS=16)
  train_t4.py       — T4: pipeline parallelism (PiPPy, 8 stages)
  train_t5.py       — T5: gradient checkpointing (6.71B primary, 3.37B fallback)
  train_t6.py       — T6: FSDP + CPU offload (ZeRO-3 equivalent)
  infer_i3.py       — I3: high-throughput vLLM (tensor_parallel_size=8)
  infer_i4.py       — I4: speculative decoding (draft + verifier per GPU)
  train_e1.py       — E1: power-capped training (22% TDP, robust restore)
  run_e2.py         — E2: cover traffic orchestrator (train 4-7 + infer 0-3)
  train_e3.py       — E3: intermittent training (30s on / 10s off)
  train_e4.py       — E4: PCIe-only allreduce (NVLink disabled)
  train_e5.py       — E5: smoothed allreduce (Ring, 128MB buffer, 1 channel)
  baseline_b1.py    — B1: idle with Llama-3.1-8B loaded
```

**New reference files:**
```
hardware_notes.md           — A100 vs H100 comparison + switching checklist
plans/                      — implementation plans for all conditions (17 files)
```

**Key changes to existing files:**
- `collect_telemetry.py`: added `gpu_model` column to CSVs for hardware tagging
- `infer_i2.py`: added `TELEMETRY_DISABLED` env var support for E2 orchestration
- `CLAUDE.md`: hardware specs now cover both A100 and H100
- All hardcoded A100 values replaced with TDP-relative references

### Design patterns

- **Thin wrappers** (E4, E5): import model from T1, set env vars, override duration/output
- **Clean forks** (T2, T3, T5, E3): copy T1 with specific modifications
- **Orchestrators** (E2): subprocess management with single telemetry collector
- **External libs** (I3, I4): vLLM for tensor-parallel inference, HF assisted generation
- **Safety guards** (E1): try/finally + atexit + signal handlers for power limit restoration

### Skipped conditions

- **I1** (batched forward pass) — bracketed by I2 + I3, low info value
- **B2** (checkpoint I/O) — short transient, won't affect sustained-signal detectors
- **B3** (CUDA graph warmup) — same reasoning as B2

### Next steps

Run the implemented workloads and collect telemetry. See priority order above. After collection, build a comparison analysis notebook covering all conditions.

---

## Session 4 — Full Telemetry Collection (2026-04-01)

### What we built

Replaced the pynvml telemetry backend with DCGM at 10Hz. Collected telemetry for 11 of the 15 planned conditions (all on H100 SXM5).

**Key change — `collect_telemetry.py` rewrite:**
- Backend: DCGM Python bindings (`/usr/local/dcgm/bindings/python3/`) instead of pynvml
- Sample rate: **10Hz** (was 1Hz) — captures sub-second spikes and the allreduce heartbeat at much finer resolution
- **20 columns** (was 14):
  - Added: `sm_active`, `tensor_active`, `dram_active`, `fp16_active`, `nvlink_tx_bytes_s`, `nvlink_rx_bytes_s`, `pcie_tx_bytes_s`, `pcie_rx_bytes_s`, `throttle_reasons`, `energy_mj`
  - Dropped: `clock_sm_mhz`, `clock_mem_mhz`, `pcie_tx_kib`, `pcie_rx_kib`
- Same public API — no workload script changes needed to adopt the new collector

**DCGM must be running before any workload:**
```bash
nv-hostengine   # start if not already running
dcgmi discovery -l  # verify it sees all 8 GPUs
```

### Workloads collected (H100, DCGM 10Hz)

| Condition | Rows | Notes |
|---|---|---|
| T1 | 26K | Large DDP pre-training (3.37B) — recollected at 10Hz |
| T2 | 25K | Small pre-training (136M) |
| T3 | 27K | Gradient accumulation (16 steps) |
| T4 | 64K | Pipeline parallelism (PiPPy, 8 stages) |
| T5 | 30K | Gradient checkpointing (6.71B, fallback 3.37B) |
| T6 | 27K | FSDP + CPU offload (ZeRO-3) |
| E3 | 55K | Intermittent training (30s on / 10s off) |
| E4 | 51K | PCIe-only allreduce (NVLink disabled) |
| I2 | 38K | Streaming autoregressive inference (Llama-3.1-8B) |
| I3 | 40K | High-throughput vLLM (8-GPU TP) |
| B1 | 26K | Idle with Llama-3.1-8B loaded |

### Shelved (need fixes before re-running)

- **E1** — RunPod blocks `nvmlDeviceSetPowerManagementLimit` — power limit control not available to tenants. Needs a different evasion approach (e.g. SM clock cap via `nvidia-smi -ac`).
- **E2** — Cover traffic orchestrator: `train_t1.py` TELEMETRY_DISABLED fix applied and DCGM duplicate group key fixed, but HF token propagation to subprocess env still broken. Fix: explicitly pass `HF_TOKEN` in `os.environ` before `subprocess.Popen`.
- **E5** — NCCL timeout with Ring/128MB buffer even at 4 channels. Likely too large a buffer for the model size — try smaller model (136M T2 config) or reduce buffer to 16MB.
- **I4** — `transformers 4.57` meta tensor error when loading 2 models per GPU in threads. Draft model swapped from gated `Llama-3.2-1B` to `Qwen/Qwen3-0.6B` (ungated), but the meta-device init bug persists. Needs investigation.

### Bug fixes made this session

- **T4** (`train_t4.py`): cast model to bf16 before PiPPy trace — was hitting fp32/bf16 dtype mismatch in pipeline stages
- **T5** (`train_t5.py`): added `optimizer.step()` to OOM probe sequence — was only catching model-creation OOM, not optimizer-state OOM
- **`train_t1.py`**: added `TELEMETRY_DISABLED` env check so E2 orchestrator doesn't overwrite T1's CSV
- **`collect_telemetry.py`**: PID-suffixed DCGM group names to avoid duplicate key errors when restarting quickly
- **`infer_i3.py`**: removed deprecated `max_seq_len_to_capture` for vLLM 0.18.1
- **`infer_i4.py`**: swapped draft model from gated `meta-llama/Llama-3.2-1B` to ungated `Qwen/Qwen3-0.6B`

### Installed packages (new this session)

```
datacenter-gpu-manager   — DCGM 3.3.9 (apt)
nvidia-ml-py3            — pynvml (still needed by E1 for power limit attempts)
vllm==0.18.1             — upgraded from 0.6.3.post1
transformers==4.57.6
pandas
```

### Key observation

`fp16_active` is always `0.000` across all conditions. Models use bf16, and DCGM field 1008 (`DCGM_FI_PROF_PIPE_FP16_ACTIVE`) tracks FP16 tensor core usage specifically. BF16 activity appears in `tensor_active` instead.

### Running the workloads (updated notes)

```bash
# DCGM must be running first
nv-hostengine

# HF_TOKEN must be set for inference workloads — Llama-3.1-8B is now gated
export HF_TOKEN=<your_token>

# Training variants
torchrun --nproc_per_node=8 workloads/train_t1.py   # T1 — large DDP pre-training (3.37B)
torchrun --nproc_per_node=8 workloads/train_t2.py   # T2 — small DDP pre-training (136M)
torchrun --nproc_per_node=8 workloads/train_t3.py   # T3 — gradient accumulation (16 steps)
torchrun --nproc_per_node=8 workloads/train_t4.py   # T4 — pipeline parallelism (PiPPy, 8 stages)
torchrun --nproc_per_node=8 workloads/train_t5.py   # T5 — gradient checkpointing (6.71B, fallback 3.37B)
torchrun --nproc_per_node=8 workloads/train_t6.py   # T6 — FSDP + CPU offload (ZeRO-3)

# Inference variants
python workloads/infer_i2.py                         # I2 — streaming autoregressive (Llama-3.1-8B) [needs HF_TOKEN]
python workloads/infer_i3.py                         # I3 — high-throughput vLLM 0.18.1 (8-GPU TP) [needs HF_TOKEN]
python workloads/infer_i4.py                         # I4 — speculative decoding [SHELVED — meta tensor bug]

# Evasion conditions
torchrun --nproc_per_node=8 workloads/train_e1.py   # E1 — power-capped training [SHELVED — RunPod blocks power limits]
python workloads/run_e2.py                           # E2 — cover traffic [SHELVED — HF token propagation]
torchrun --nproc_per_node=8 workloads/train_e3.py   # E3 — intermittent training (30s on / 10s off)
torchrun --nproc_per_node=8 workloads/train_e4.py   # E4 — PCIe-only allreduce (NVLink disabled)
torchrun --nproc_per_node=8 workloads/train_e5.py   # E5 — smoothed allreduce [SHELVED — NCCL timeout]

# Baselines
python workloads/baseline_b1.py                      # B1 — idle with Llama-3.1-8B loaded [needs HF_TOKEN]
```

### Next steps

Build multi-condition comparison analysis notebook covering all 11 collected conditions. Fix shelved workloads (E1, E2, E5, I4) if time/budget allows.

---

## Output files to save before stopping the pod

**From session 1 (baseline exploration):**
- `telemetry_baseline.csv`
- `telemetry_dashboard.png`, `telemetry_heatmap.png`, `nvlink_topology.png`
- `nvlink_per_link.png`, `nvlink_rate_timeline.png`

**From session 2 (mock workloads):**
- `data/t1_telemetry.csv`, `data/i2_telemetry.csv`
- `comparison_timeseries.png`, `comparison_distributions.png`
- `comparison_heatmaps.png`, `comparison_power_variability.png`
- `notebooks/comparison.ipynb`

**From session 4 (full collection, DCGM 10Hz, H100):**
- `data/t1_telemetry.csv` through `data/b1_telemetry.csv` (11 files)
- `data/e2_telemetry.csv`, `data/e5_telemetry.csv` (partial/debug runs)

Copy off before stopping. Simplest: `scp` or VS Code file explorer drag-and-drop.

---

## Claude memory — setup on a new node

Memory files are stored in `memory/` in this repo (committed to GitHub) so they survive pod restarts.

On a fresh node, restore them with:
```bash
mkdir -p /root/.claude/projects/-root-gpu-telemetry-exploration/memory
cp memory/* /root/.claude/projects/-root-gpu-telemetry-exploration/memory/
```

Run this before starting a session so context (project state, Atharva's preferences, etc.) is available.

---

## Atharva's context and preferences

- New to GPU telemetry tooling (first time using DCGM/pynvml). Has strong conceptual background on GPU architecture, NVSwitch, NVLink, training vs inference differences.
- Prefers interactive Jupyter over scripts. Explore first, structure later.
- Informal communication, concise responses. Explicit uncertainty labeling appreciated.
- This session is pure exploration — no deliverable, no deadline. Follow curiosity.
- If something unexpected shows up in the data (outlier GPU, unexpected NVLink traffic, weird power signature), dig into it.

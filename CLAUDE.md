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

| Item | Value |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB |
| GPU count | 8 |
| NVLink version | NVLink 3.0 |
| NVLink ports per GPU | 12 (L0–L11) |
| NVLink bandwidth | 600 GB/s bidirectional per GPU |
| HBM2e per GPU | 80 GB |
| GPU TDP | ~400W |
| NVSwitch | 4 chips on HGX baseboard (all 8 GPUs fully connected) |
| Inter-GPU topology | All-to-all within baseboard via NVSwitch |
| Scale-out | InfiniBand (not relevant for this single-node session) |

**On an idle node, expect roughly:**
- Power: 50–150W per GPU
- SM utilization: ~0%
- NVLink traffic: ~0 (maybe a few MB of control-plane)
- Memory: whatever is loaded (should be near-zero if nothing's running)

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

# Find available NVLink field IDs (should see 409-420 for A100)
dcgmi fieldids -l | grep -i nvlink
```

**Key DCGM field IDs for A100:**
- `203` = SM utilization (%)
- `155` = Power draw (W)
- `150` = Temperature (°C)
- `100` / `101` = Framebuffer free / used (MiB)
- `409–420` = NVLink bandwidth L0–L11 (cumulative bytes per link — diff consecutive readings for rate)

**Note**: Fields 409–420 are **cumulative counters**, not rates. To get GB/s, diff two readings and divide by the interval.

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
- Compute stress (matmul loop): should spike power to ~400W/GPU, SM util to ~100%, NVLink stays near zero
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
workload_plan.md            — detailed design decisions for T1 + I2 scripts
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

### Key findings

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
```

### Running the workloads

```bash
# T1 — pre-training
torchrun --nproc_per_node=8 workloads/train_t1.py

# I2 — inference (no HF login needed, model is public)
python workloads/infer_i2.py
```

### Next up — evasion conditions

Priority order from `mock_conditions.md`:
1. **E1** — power-capped training (`nvmlDeviceSetPowerManagementLimit` to ~150W)
2. **E4** — PCIe-only allreduce (`NCCL_P2P_DISABLE=1`) — kills NVLink signal entirely
3. **T3** — gradient accumulation (reduces allreduce frequency)
4. **I3** — high-throughput batched inference (hardest inference case to distinguish from training)

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

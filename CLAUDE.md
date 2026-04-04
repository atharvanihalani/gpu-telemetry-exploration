# GPU Telemetry Exploration — Handoff for Claude Code

## What we're doing and why

Atharva is building a trusted third-party org to verify that frontier AI labs have actually paused pre-training runs during a coordinated AI slowdown. Core problem: how do you tell, from outside, whether a GPU cluster is secretly training vs sitting idle or running inference?

This is **open-ended exploration** — collecting raw GPU telemetry under controlled workloads to understand what signals distinguish training from inference, and how robust detection is under adversarial evasion.

**Detailed session-by-session history:** see `SESSION_LOG.md`

---

## Current state (after session 8, 2026-04-03)

### Data collected (14 conditions, H100 SXM5, DCGM 10Hz)

| Category | Conditions | Status |
|---|---|---|
| Training | T1 (DDP 3.37B), T2 (DDP 136M), T3 (grad accum), T4 (pipeline), T5 (grad ckpt), T6 (FSDP) | All collected |
| Inference | I2 (autoregressive), I3 (vLLM 8-GPU TP), I4 (speculative decoding) | All collected |
| Evasion | E2 (cover traffic), E3 (intermittent), E4 (PCIe-only), E5 (smoothed allreduce) | All collected |
| Baseline | B1 (idle + model loaded) | Collected |
| **Shelved** | E1 (power cap — deprioritized, unrealistic evasion) | Low priority |

### Classifier (3 rules, OR logic, 12/14 correct)

| Rule | Threshold | Physical basis |
|---|---|---|
| `mean_power` | > 400W | Sustained high compute |
| `tensor_sm_ratio` | > 0.25 | Matmul-dominated workload (tensor_active / sm_active) |
| `nvlink_autocorr_peak` | > 0.3 | Periodic NVLink heartbeat (autocorrelation in 0.2-5s lag) |

**Known issues:**
- **I3 false positive**: vLLM tensor-parallel inference triggers nvlink_autocorr (0.699) — periodic activation exchanges look like allreduce. Needs compound rule (autocorr AND power).
- **E4 miss**: PCIe-only allreduce kills NVLink signal, power stays moderate (~300W), tensor_ratio borderline. No current rule catches it.
- **E2 near miss**: aggregate tensor_sm_ratio is 0.242 (threshold 0.25) — cover traffic nearly dodges it. Caught by power (408W) and nvlink_autocorr instead.

### Key insight

Every evasion breaks ONE signal but leaves others intact. Multi-signal OR-logic detection is robust — no evasion defeats all rules simultaneously.

---

## Hardware specs

We flip between A100 and H100 nodes. Both are 8-GPU, 80GB, all-to-all NVSwitch.

| Item | A100 SXM4 | H100 SXM5 |
|---|---|---|
| NVLink | 3.0, 12 ports, 600 GB/s | 4.0, 18 ports, 900 GB/s |
| GPU TDP | ~400W | ~700W |
| Max SM clock | ~1410 MHz | ~1980 MHz |

**Idle baseline:** ~60-80W, ~0% SM, ~0 NVLink. See `hardware_notes.md` for full comparison.

---

## Why these signals matter

| Signal | Training | Inference | Detection value |
|---|---|---|---|
| **Power** | Flat, sustained, high | Spiky, demand-following | Hardest to hide |
| **NVLink** | Periodic symmetric bursts (allreduce) | Sparse/asymmetric | Smoking gun — heartbeat pattern |
| **SM util** | Relentlessly high | Bursty | Training is compute-bound |
| **Memory** | Very high (weights + grads + optimizer) | Lower (weights + KV cache) | Optimizer states don't exist in inference |

**Core crux:** temporal pattern over time, not instantaneous snapshot.

---

## Running workloads

```bash
# Fresh node setup
apt-get update -q && apt-get install -y datacenter-gpu-manager
nv-hostengine
pip install nvidia-ml-py3 pandas matplotlib seaborn transformers accelerate huggingface_hub
export HF_TOKEN=<token>  # or add to .env

# Training
torchrun --nproc_per_node=8 workloads/train_t1.py   # T1 — large DDP (3.37B)
torchrun --nproc_per_node=8 workloads/train_t2.py   # T2 — small DDP (136M)
torchrun --nproc_per_node=8 workloads/train_t3.py   # T3 — gradient accumulation
torchrun --nproc_per_node=8 workloads/train_t4.py   # T4 — pipeline parallelism
torchrun --nproc_per_node=8 workloads/train_t5.py   # T5 — gradient checkpointing
torchrun --nproc_per_node=8 workloads/train_t6.py   # T6 — FSDP + CPU offload

# Inference
python workloads/infer_i2.py                         # I2 — autoregressive [needs HF_TOKEN]
python workloads/infer_i3.py                         # I3 — vLLM 8-GPU TP [needs HF_TOKEN]
python workloads/infer_i4.py                         # I4 — speculative decoding [needs HF_TOKEN]

# Evasion
python workloads/run_e2.py                           # E2 — cover traffic (train+infer split)
torchrun --nproc_per_node=8 workloads/train_e3.py   # E3 — intermittent (30s on / 10s off)
torchrun --nproc_per_node=8 workloads/train_e4.py   # E4 — PCIe-only allreduce
torchrun --nproc_per_node=8 workloads/train_e5.py   # E5 — smoothed allreduce (Ring, 32MB, 2ch)

# Baseline
python workloads/baseline_b1.py                      # B1 — idle + model loaded [needs HF_TOKEN]
```

---

## DCGM reference

**Key field IDs:** 155 (power), 150 (temp), 252 (mem used), 203 (gpu util), 1002 (sm_active), 1004 (tensor_active), 1005 (dram_active), 1011/1012 (nvlink tx/rx bytes/s), 112 (throttle), 156 (energy).

`fp16_active` (1008) is always 0 — models use bf16, which shows in `tensor_active`.

NVLink fields 409-420 (A100, 12 links) or 409-426 (H100, 18 links) are cumulative counters — diff for rate.

---

## Telemetry CSV schema (20 columns)

`timestamp, phase, gpu, gpu_model, power_w, temp_c, mem_used_mib, mem_total_mib, gpu_util_pct, mem_util_pct, sm_active, tensor_active, dram_active, fp16_active, pcie_tx_bytes_s, pcie_rx_bytes_s, nvlink_tx_bytes_s, nvlink_rx_bytes_s, throttle_reasons, energy_mj`

---

## Next steps

- Rethink classifier: compound rules for I3 false positive, possibly PCIe traffic rule for E4
- Threshold sensitivity sweep
- Write up findings for verification proposal
- Minor: T2 is missing the `TELEMETRY_DISABLED` env var check that T1 has (would matter if T2 is ever used in an orchestrator like E2)

---

## Claude memory — setup on a new node

```bash
mkdir -p /root/.claude/projects/-root-gpu-telemetry-exploration/memory
cp memory/* /root/.claude/projects/-root-gpu-telemetry-exploration/memory/
```

## Timestamp hook

`.claude/settings.json` has a `UserPromptSubmit` hook injecting current time. See `SESSION_LOG.md` session 7 for the JSON structure.

---

## Atharva's context and preferences

- Strong conceptual background on GPU architecture, NVSwitch, NVLink, training vs inference. New to DCGM/pynvml tooling.
- Prefers interactive Jupyter over scripts. Explore first, structure later.
- Informal communication, concise responses. Dig into anomalies.
- Open-ended exploration — follow curiosity, no deliverable deadline.

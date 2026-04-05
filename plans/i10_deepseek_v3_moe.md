# I10 — DeepSeek V3 MoE Inference (2 nodes, 16× H100)

## Overview

First multi-node inference condition. DeepSeek V3 (671B params, 256 experts, top-2 routing, MLA attention) in FP8 across 2 nodes. Direct comparison target: T14 (TP+EP+DP MoE training).

**Core question:** Does multi-node MoE inference produce IB/NVLink patterns that could be confused with MoE training?

## Framework: SGLang (offline batch mode)

**Why SGLang:**
- Native DeepSeek V3 support (validated by DeepSeek team)
- Multi-node EP via `--dp` flag (internally implements EP for MoE layers)
- FP8 weight loading (DeepSeek V3 ships native FP8 weights)
- Continuous batching built in
- Framework diversity (I3 uses vLLM)

**Why offline mode:** Simpler to orchestrate, sufficient for telemetry. Server mode deferred to future re-run (see TODO).

**Fallback:** vLLM with `--tensor-parallel-size 8 --pipeline-parallel-size 2` (TP+PP instead of TP+EP — different but still multi-node).

## Parallelism layout: TP=8 within node, EP=2 across nodes

```
Node 0 (GPUs 0-7):                    Node 1 (GPUs 8-15):
  TP=8 across GPUs 0-7                  TP=8 across GPUs 8-15
  (attention + dense layers)             (attention + dense layers)

  128 of 256 experts reside here         128 of 256 experts reside here
        |                                       |
        └──── EP all-to-all over IB ────────────┘
         (token shuffle for expert dispatch)
```

- TP=8 within node for dense/attention layers (NVLink all-reduce)
- EP=2 across nodes: 128 experts per node, 16 per GPU. All-to-all token shuffle over IB.
- Shared expert (always activated) replicated on every GPU
- **No DP** — inference has no gradient synchronization (key structural difference from T14)

## Memory budget

| Component | Per-GPU estimate | Notes |
|---|---|---|
| Model weights (FP8) | ~42 GB | 671B / 16 GPUs × 1 byte |
| KV cache | ~20-30 GB | MLA compresses KV ~2x vs standard MHA |
| Activations + buffers | ~5 GB | Forward-only, no gradients/optimizer |
| **Total** | **~67-77 GB** | Fits in 80 GB. Set `mem_fraction_static=0.88` |

## Load generation

Offline batch mode — feed prompts continuously for 5 minutes:

- **Prompt length:** 128 tokens input, 256 tokens output
- **Batch size:** 32-64 concurrent sequences (tune based on memory)
- **`ignore_eos=True`** to ensure consistent output length
- **Prompts:** random topic+subject pattern (from I3), content doesn't affect telemetry

## Expected telemetry signatures

| Signal | T14 (TP+EP+DP training) | I10 (TP+EP inference) | Why different |
|---|---|---|---|
| NVLink pattern | Continuous TP + variable EP | **Same** — continuous TP + variable EP | Same TP+EP within node |
| NVLink volume | ~32 GB/s | **Lower** (~15-25 GB/s) | Forward-only, no backward TP all-reduces |
| IB pattern | **Periodic** (DP gradient allreduce) | **Variable, non-periodic** (EP only) | **KEY DIFFERENTIATOR** — no gradient allreduce |
| IB volume | **79 GB/s** (massive DP allreduce) | **5-15 GB/s** (EP token shuffles only) | No gradient allreduce in inference |
| IB autocorrelation | **High** (periodic) | **Low** (demand-driven) | Smoking gun for detection |
| Power | 480W sustained | **250-400W** | Decode is memory-bound, no backward pass |
| tensor_sm_ratio | 0.369 | **0.1-0.25** | Forward-only, less matmul dominance |
| Memory | Very high (weights+grads+optimizer) | **Moderate** (weights+KV only) | No optimizer states |

**Classifier predictions:**

| Rule | T14 | I10 (predicted) |
|---|---|---|
| `mean_power > 400W` | 480W ✓ triggers | **Borderline** (~300-400W), likely below |
| `tensor_sm_ratio > 0.25` | 0.369 ✓ triggers | **Likely below** (~0.15-0.25) |
| `nvlink_autocorr > 0.3` | Low, no trigger | **Low**, no trigger |

I10 should be correctly classified as non-training. The interesting new signal: **IB periodicity and volume** can distinguish MoE inference from MoE training even when NVLink patterns are similar.

## Collection procedure

Three collectors on each node (same as T10-T15):

| Collector | Output | Rate |
|---|---|---|
| DCGM | `data/i10_node{N}_telemetry.csv` | 10 Hz |
| IB | `data/i10_node{N}_ib.csv` | 10 Hz |
| BMC | `data/i10_node{N}_bmc.csv` | 2s |

**Phases:** `loading` → `warmup` (30s) → `steady` (~270s) → `cooldown` (5s)

**Architecture:** Collectors run as threads in the launcher script (same process, started before SGLang engine init). DCGM connects to nv-hostengine, IB reads sysfs, BMC reads IPMI — all independent of the inference framework.

## Script architecture

Single script `workloads/infer_i10.py`, launched on each node:

```
infer_i10.py
├── Config (model, durations, output paths)
├── main():
│   ├── Detect node rank (from args or env)
│   ├── Start 3 collectors (DCGM + IB + BMC) — phase "loading"
│   ├── Initialize SGLang engine (TP=8, DP=2 for EP)
│   ├── Warmup (30s of generation) — phase "warmup"
│   ├── Steady-state generation loop (~270s) — phase "steady"
│   ├── Cooldown (5s) + collector shutdown
│   └── os._exit(0) if needed
```

## Launch

```bash
# Both nodes need: pip install "sglang[all]"
# Both nodes need DeepSeek V3 downloaded to HF cache (~350 GB)

# Node 0:
python workloads/infer_i10.py --node-rank 0 --master-addr <NODE0_PRIVATE_IP>

# Node 1:
python workloads/infer_i10.py --node-rank 1 --master-addr <NODE0_PRIVATE_IP>
```

## Setup steps

1. Install SGLang on both nodes
2. Download DeepSeek V3 FP8 on both nodes (no shared filesystem — ~350 GB each)
3. Test single-node SGLang with DeepSeek V3 (TP=8 on one node)
4. Test multi-node (2-node EP)
5. Full 5-minute collection run

## Pitfalls

1. **No shared filesystem** — model must be downloaded on both nodes independently
2. **Memory pressure** — 42 GB/GPU in weights leaves ~35 GB for KV cache. May need to limit concurrent sequences.
3. **SGLang manages its own processes** — unlike torchrun, SGLang spawns workers internally. Collectors must be started in the launcher process before engine init.
4. **NCCL IB env vars** — may need `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME` (same as T10-T15)
5. **`os._exit(0)`** — SGLang may not clean up gracefully on multi-node

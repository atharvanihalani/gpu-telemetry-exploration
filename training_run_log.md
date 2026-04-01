# Workload Run Log (2026-04-01)

Hardware: 8x NVIDIA H100 80GB HBM3
Collector: DCGM backend, 10Hz, 20 columns

## Collected

| Workload | Duration | Rows | Key stats (steady-state) |
|---|---|---|---|
| T1 | 305s | 26K | 693 steps, 3.37B DDP |
| T2 | 305s | 25K | 8053 steps, 136M DDP |
| T3 | 306s | 27K | 52 opt steps (16x accum), 3.37B |
| T4 | ~300s + hang | 64K | 980 steps, PiPPy pipeline, loss=NaN |
| T5 | 305s | 30K | 575 steps, 3.37B fallback w/ grad ckpt |
| T6 | 308s | 27K | 85 steps, FSDP+CPU offload |
| E3 | ~305s + hang | 55K | 7 duty cycles (30s on / 10s off) |
| E4 | 608s | 51K | 135 steps, PCIe-only allreduce |
| I2 | ~300s | 38K | Llama-3.1-8B, 8 independent streams |
| I3 | ~300s | 40K | Llama-3.1-8B, vLLM 0.18.1, 8-GPU TP, 11.5K tok/s |
| B1 | ~300s | 26K | Llama-3.1-8B loaded, idle, 120W/0% SM |

## Shelved / Failed

| Workload | Issue |
|---|---|
| E1 | RunPod blocks `nvmlDeviceSetPowerManagementLimit` — needs bare-metal |
| E2 | HF token + DCGM duplicate key issues; needs careful subprocess env setup |
| E5 | NCCL timeout even with 4 channels — 3.37B too large for Ring/128MB config |
| I4 | transformers 4.57 meta tensor error when loading 2 models/GPU in threads |

## Errors (full details)

### T3 — first attempt: EADDRINUSE
Stale torchrun TCP store from T2 hadn't released the port yet. Killed orphan processes and retried successfully.

### T4 — PipeliningShapeError (fixed), then hung on exit
bf16/fp32 mismatch fixed (cast model before PiPPy trace). Training ran but processes hung on cleanup — killed manually. Data usable.

### T5 — OOM at optimizer.step() (fixed)
6.71B OOM'd at first optimizer step. Fixed by adding `optimizer.step()` to probe sequence. Falls back to 3.37B.

### E1 — Insufficient Permissions
RunPod doesn't grant `nvmlDeviceSetPowerManagementLimit`. Even `nvidia-smi -pl 200` blocked.

### E2 — multiple issues
1. `ModuleNotFoundError: transformers` (installed)
2. DCGM duplicate key (fixed with PID-suffixed group names)
3. HF token not propagating to inference subprocess (Llama-3.1-8B gated)
4. train_t1.py overwrote T1 CSV (fixed: added TELEMETRY_DISABLED check)

### E5 — NCCL timeout
Ring + 128MB buffer + 1ch: single allreduce of 13.5GB took 10+ min. Bumped to 4ch, still timed out. Needs smaller model or more channels.

### I4 — meta tensor error with transformers 4.57
vLLM install upgraded transformers. Loading 2 models per GPU in threads triggers "Cannot copy out of meta tensor". Needs version pin or sequential loading.

### B1/I4 overlap — trimmed
28s overlap where I4 was loading while B1 was in steady/cooldown. B1 data trimmed to pre-overlap timestamps. Impact was minimal (120W vs 121.9W).

# T15 — Full FSDP (ZeRO-3) Multi-Node Training

## Overview

Full FSDP across all 16 GPUs on 2 nodes. Every parameter sharded into 16 pieces. Per-layer all-gathers (forward) and reduce-scatters (backward) spanning both NVLink and IB.

First condition with continuous, non-periodic IB traffic. Unlike DDP (T10) where IB carries one allreduce at step boundaries, FSDP has IB active throughout the entire step — every layer triggers collectives over IB.

## Architecture decisions

### Import model from T1
Same 3.37B GPT as T1/T10/T11. Import `GPT` and `TransformerBlock` directly — no model duplication.

### FSDP wrapping
- `transformer_auto_wrap_policy` with `TransformerBlock` — each block is its own FSDP unit
- `ShardingStrategy.FULL_SHARD` (ZeRO-3) — entire point of T15
- `BackwardPrefetch.BACKWARD_PRE` — overlap communication with compute
- `sync_module_states=True` — rank 0 broadcasts weights during init
- `MixedPrecision` all bf16 (param, reduce, buffer)

### NO CPU offload
Unlike T6 (single-node FSDP + CPU offload), T15 deliberately omits CPU offloading to isolate the pure FSDP communication pattern without PCIe noise.

### NO autocast
FSDP's `MixedPrecision` policy handles dtype casting — `torch.amp.autocast` is redundant. Matches T6.

### Standard cleanup
FSDP does not use composable `parallelize_module`, so `destroy_process_group()` should work. **UPDATE: it hangs anyway with multi-node FSDP. Needs `os._exit(0)` workaround like T11+.**

## Memory math
- 3.37B params, 16-way sharding → ~0.81 GB allocated per GPU after setup
- Peak with activations: ~3.75 GB per GPU (observed)
- Trivially fits on 80 GB H100

## Expected telemetry signature

| Signal | T15 (FSDP 16-GPU) | T10 (DDP) | T11 (TP+DP) |
|---|---|---|---|
| NVLink | 23 GB/s continuous | 18 GB/s periodic | 56 GB/s continuous |
| IB | **25.6 GB/s continuous** | moderate periodic | moderate periodic |
| Power | 417W | ~500W | ~508W |
| tensor_sm_ratio | **0.657** (highest!) | ~0.40 | ~0.47 |
| GPU memory | 3.75 GB | ~38 GB | — |

Key: IB is **continuous and non-periodic** — first condition with this pattern.

## Config
Same as T1: D_MODEL=3072, N_LAYERS=28, N_HEADS=24, BATCH_SIZE=4 per GPU.

## Output files
- `data/t15_node{N}_telemetry.csv` (DCGM, 10Hz)
- `data/t15_node{N}_ib.csv` (InfiniBand, 10Hz)
- `data/t15_node{N}_bmc.csv` (BMC, 2s)

## Launch
```bash
# Node 0:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
  --master_addr=192.168.241.135 --master_port=29500 \
  workloads/train_t15.py

# Node 1:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
  --master_addr=192.168.241.135 --master_port=29500 \
  workloads/train_t15.py
```

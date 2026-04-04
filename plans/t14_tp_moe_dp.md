# T14 — TP + EP + DP Hybrid MoE Training

## Overview

Most realistic frontier MoE training configuration. Combines:
- **TP (8-way, within node):** Shards attention and dense FFN layers across 8 GPUs via NVLink
- **EP (8-way, within node):** All-to-all token shuffle for MoE FFN layers, same 8 GPUs
- **DP (2-way, across nodes):** Gradient allreduce over InfiniBand

Model architecture: same alternating dense/MoE blocks as T12 (odd blocks are MoE), but with TP on the dense parts (attention + dense FFN).

## Architecture decision: parallelize_module + replicate + manual EP

Use `parallelize_module` for TP, `replicate()` for DP, manual `dist.all_to_all_single` for EP.

**Why:**
- T11 validates `parallelize_module` + `replicate()` composition
- `replicate()` handles DTensor localization → no `foreach=False` needed in AdamW
- MoE all-to-all is independent of both TP and DP
- `find_unused_parameters=True` in `replicate()` as safety for zero-token expert edge case

## TP sharding scope

```python
for i, block in enumerate(model.blocks):
    # Attention: ALWAYS TP-sharded
    parallelize_module(block.attn, tp_mesh, {
        "qkv": ColwiseParallel(),
        "proj": RowwiseParallel(),
    })
    # Dense FFN: only even blocks
    if i % 2 == 0:
        parallelize_module(block.ffn, tp_mesh, {
            "0": ColwiseParallel(),
            "2": RowwiseParallel(),
        })
    # Odd blocks: MoE FFN handled by EP, NOT TP-sharded
```

**NOT sharded:** MoE router, MoE expert, embeddings, LayerNorm, output head.

## Process groups

```
DeviceMesh: (dp=2, tp=8) — same as T11
EP groups: manual dist.new_group per node — same as T12
```

EP group has same GPU membership as TP mesh but is a separate NCCL communicator.

## Application order

1. Build model on device in bf16
2. Apply TP (`parallelize_module`) to attention + dense FFN
3. Apply DP (`replicate(model, device_mesh=dp_mesh, find_unused_parameters=True)`)
4. Create optimizer (default `foreach=True` is fine)

## Expected telemetry signature

| Signal | T11 (TP+DP) | T12 (EP+DP) | T14 (TP+EP+DP) |
|---|---|---|---|
| NVLink pattern | Continuous (TP) | Variable (all-to-all) | **Continuous + variable bursts overlaid** |
| NVLink volume | ~56 GB/s | ~20-40 GB/s | **Highest** (~60-80 GB/s est.) |
| NVLink autocorr | Low (continuous) | Low (variable) | **Low** (no periodicity) |
| IB traffic | Periodic allreduce | Periodic allreduce | **Periodic allreduce** |
| Power | ~508W | ~500-600W | **~500-600W** |

## Config

Same model as T1/T11 (3072 d_model, 28 layers, 24 heads). N_EXPERTS=8, TOP_K=2 (from T12). BATCH_SIZE=4 per GPU.

## Pitfalls

1. **MoE bf16 casting before comms** — autocast doesn't cover dist ops
2. **TP-aware attention** — must use T11's CausalSelfAttention (infers local_heads)
3. **Don't TP-shard MoE layers** — router and expert stay full-size per GPU
4. **Process group creation order** — all ranks must call new_group in same order
5. **`os._exit(0)`** — same TP cleanup hang as T11/T13

## Output files

- `data/t14_node{N}_telemetry.csv` (DCGM, 10Hz)
- `data/t14_node{N}_ib.csv` (InfiniBand, 10Hz)
- `data/t14_node{N}_bmc.csv` (BMC, 2s)

## Launch

```bash
# Node 0:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
  --master_addr=192.168.241.135 --master_port=29500 \
  workloads/train_t14.py

# Node 1:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
  --master_addr=192.168.241.135 --master_port=29500 \
  workloads/train_t14.py
```

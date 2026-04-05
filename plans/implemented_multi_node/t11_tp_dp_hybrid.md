# T11 — TP+DP Hybrid Training (8-way TP within node, 2-way DP across nodes)

## Context

T10 is pure DP — allreduce over NVLink produces a periodic heartbeat our classifier catches easily. But frontier training uses TP within each node (continuous NVLink) + DP across nodes (periodic IB). This is the configuration that most threatens our NVLink autocorrelation rule, because intra-node NVLink shows continuous traffic, not periodic bursts.

T11 produces the realistic frontier training telemetry signature and tests whether our classifier still works.

## Approach

Use PyTorch's native `parallelize_module` + `DeviceMesh` (all available in 2.6):
- `ColwiseParallel` for QKV projections and FFN up-projections
- `RowwiseParallel` for output projections and FFN down-projections
- `init_device_mesh("cuda", (2, 8), ("dp", "tp"))` for the 2D mesh
- `replicate` (composable DDP) on the DP dimension

## Script: `workloads/train_t11.py`

~150 lines. Reuses T1's model classes + T10's multi-node telemetry pattern.

### Key structure

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module, ColwiseParallel, RowwiseParallel,
)
from torch.distributed._composable.replicate import replicate

# 2D mesh: (dp=2, tp=8) — TP within node, DP across nodes
mesh = init_device_mesh("cuda", (n_nodes, gpus_per_node), ("dp", "tp"))
tp_mesh = mesh["tp"]
dp_mesh = mesh["dp"]

# Build model on local GPU
model = GPT(...).to(device).to(torch.bfloat16)

# Apply TP sharding to every transformer block
for block in model.blocks:
    parallelize_module(block.attn, tp_mesh, {
        "qkv": ColwiseParallel(),    # shard output dim across TP group
        "proj": RowwiseParallel(),   # shard input dim, all-reduce output
    })
    parallelize_module(block.ffn, tp_mesh, {
        "0": ColwiseParallel(),      # FFN up-projection (nn.Sequential index 0)
        "2": RowwiseParallel(),      # FFN down-projection (nn.Sequential index 2)
    })

# Apply DP replication across nodes
replicate(model, device_mesh=dp_mesh)

# Training loop — identical to T1, but backward triggers:
#   - TP all-reduces within node (NVLink, every layer)
#   - DP all-reduce across nodes (IB, every step)
```

### Telemetry

Same three-collector pattern as T10 (copy-paste):
- DCGM at 10Hz (per node) → `data/t11_node{N}_telemetry.csv`
- IB at 10Hz (per node) → `data/t11_node{N}_ib.csv`
- BMC at 2s (per node) → `data/t11_node{N}_bmc.csv`

### Model sizing

Same 3.2B GPT config as T1. With 8-way TP, each GPU holds 1/8th of each layer's weights (~400M params locally). Memory usage drops dramatically — plenty of headroom.

### Launch

Same as T10 — manual per-node, no SSH:

```bash
# Node 0:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
  --master_addr=192.168.242.186 --master_port=29500 \
  workloads/train_t11.py

# Node 1:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
  --master_addr=192.168.242.186 --master_port=29500 \
  workloads/train_t11.py
```

## Expected telemetry vs T10

| Signal | T10 (pure DP) | T11 (TP+DP) |
|---|---|---|
| NVLink pattern | Periodic bursts (allreduce heartbeat) | **Continuous high bandwidth** (TP layer syncs) |
| NVLink autocorr | High (~0.7+) | **Low** (no periodicity) |
| IB traffic | Periodic (DP allreduce) | **Still periodic** (DP allreduce, same) |
| Power | ~687W sustained | Similar sustained high |
| Tensor/SM ratio | ~0.47 | Similar (still matmul-dominated) |

## Classifier impact

- `nvlink_autocorr_peak > 0.3` → **will likely fail** (continuous ≠ periodic)
- `mean_power > 400W` → still triggers
- `tensor_sm_ratio > 0.25` → still triggers
- **New signal opportunity**: sustained high NVLink bandwidth (mean, not autocorrelation)

## Files to create

- `workloads/train_t11.py` — TP+DP training script with all 3 collectors
- `plans/t11_tp_dp_hybrid.md` — copy of this plan

## Verification

1. Training runs at WORLD_SIZE=16
2. NVLink TX is **sustained high** (not periodic bursts) — visually distinct from T10
3. IB traffic shows periodic DP allreduce
4. Power and tensor_ratio similar to T10
5. `nvlink_autocorr_peak` is low (below 0.3 threshold)

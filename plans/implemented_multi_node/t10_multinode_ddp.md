# T10 — Multi-Node DDP (16-GPU, 2 nodes)

## Why

All 14 conditions so far are single-node (8 GPU). Real pre-training uses multi-node clusters. T10 is the multi-node ground truth: what does 16-GPU DDP look like when allreduce spans NVLink (intra-node) + InfiniBand (inter-node)?

## What changes from T1

Almost nothing in the training logic — `torchrun` handles multi-node DDP transparently. The only real change is **telemetry collection**: DCGM is node-local, so we need a collector running on each node.

## Script: `workloads/train_t10.py`

Imports model + config from T1 (same thin-wrapper pattern as E4/E5). Key differences:

1. **Telemetry on every node**: `LOCAL_RANK == 0` starts a collector, not just global `rank == 0`
2. **Output per-node**: `data/t10_node{N}_telemetry.csv` where N = node_rank
3. **Node identification**: add `node` column to CSV (derived from `RANK // nproc_per_node`)
4. **Phase sync**: only global rank 0 prints progress; all LOCAL_RANK==0 processes mirror phase changes via a simple broadcast

### Collector changes

The collector itself (`collect_telemetry.py`) stays unchanged — it's already node-local by design. T10 just starts one per node.

```python
from workloads.train_t1 import GPT, D_MODEL, N_LAYERS, ...

DURATION_S = 5 * 60
OUTPUT_CSV_TEMPLATE = "data/t10_node{}_telemetry.csv"

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    nproc = torch.cuda.device_count()  # 8
    node_rank = rank // nproc

    # Every node's LOCAL_RANK=0 runs a collector
    collector = None
    if local_rank == 0 and not os.environ.get("TELEMETRY_DISABLED"):
        collector = TelemetryCollector(OUTPUT_CSV_TEMPLATE.format(node_rank))
        collector.start()
        collector.set_phase("warmup")

    # ... rest is identical to T1 training loop ...
    # phase changes: broadcast from rank 0, all local_rank==0 mirror
```

### Node column in CSV

Two options:
- **A)** Add `node` column to the collector itself (requires changing collect_telemetry.py API)
- **B)** Don't change collector; derive node from filename during analysis (`t10_node0` → node 0)

**Going with B** — simpler, no collector changes, and the filenames already encode it. We can add a `node` column during the merge step in the notebook.

## Launch (manual, no SSH needed)

```bash
# Node 0 (192.168.242.186):
source ~/venv/bin/activate
cd ~/gpu-telemetry-exploration
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
  --master_addr=192.168.242.186 --master_port=29500 \
  workloads/train_t10.py

# Node 1 (192.168.240.15):
source ~/venv/bin/activate
cd ~/gpu-telemetry-exploration
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
  --master_addr=192.168.242.186 --master_port=29500 \
  workloads/train_t10.py
```

Both must start within ~60s of each other (NCCL timeout).

## Expected telemetry vs T1

- **Power**: similar sustained high (~650-700W) — same compute per GPU
- **NVLink**: heartbeat still visible (intra-node allreduce), but amplitude may differ since gradient volume per-node changes with 16-way vs 8-way
- **IB traffic**: NOT visible in DCGM — would need sysfs polling (`/sys/class/infiniband/mlx5_ibN/ports/1/counters/`) as a follow-up
- **Step time**: slightly slower (IB latency for inter-node allreduce)
- **Classifier**: all 3 rules should still trigger (power, tensor_ratio, nvlink_autocorr)

## Output

```
data/t10_node0_telemetry.csv   (~24K rows, 8 GPUs, 5 min)
data/t10_node1_telemetry.csv   (~24K rows, 8 GPUs, 5 min)
```

Node 1's CSV will need to be copied to node 0 (or pushed via git) for analysis.

## Files to create/modify

- **Create**: `workloads/train_t10.py` (~80 lines — imports from T1, adds multi-node telemetry)
- **No changes**: `collect_telemetry.py`, `classifier/`, notebooks

## Verification

1. Both nodes print `[telemetry] DCGM connected — 8x NVIDIA H100 80GB HBM3`
2. Training runs at 16-GPU scale (WORLD_SIZE=16 in logs)
3. Two CSV files produced, each with ~24K rows, 8 GPUs
4. NVLink TX shows periodic heartbeat on both nodes
5. Existing classifier rules trigger on both node CSVs

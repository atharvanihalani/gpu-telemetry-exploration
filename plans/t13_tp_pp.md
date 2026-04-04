# T13 — TP+PP Hybrid Training (8-way TP within node, 2-stage PP across nodes)

## Overview

Combines 8-way Tensor Parallelism within each node with 2-stage Pipeline Parallelism across 2 nodes. No data parallelism. Each node runs one pipeline stage:

- Node 0 (stage 0): embedding + blocks 0-13 (first half), TP across 8 GPUs
- Node 1 (stage 1): blocks 14-27 + ln_f + head (second half), TP across 8 GPUs

## Expected telemetry signature

| Signal | T11 (TP+DP) | T13 (TP+PP) |
|---|---|---|
| NVLink pattern | Continuous high (TP) | **Same continuous high (TP)** |
| NVLink volume | ~56 GB/s | **Similar** (~50-56 GB/s) |
| IB traffic pattern | Periodic allreduce (DP) | **Periodic P2P** (pipeline activations) |
| IB traffic volume | High (full gradient allreduce) | **Lower** (~48 MiB per micro-batch) |
| IB symmetry | Symmetric (allreduce) | **Asymmetric** (fwd one way, bwd the other) |
| Power | ~508W | **Slightly lower** (pipeline bubble ~20%) |

Key differentiator from T10/T11: IB traffic is **point-to-point** (one sender, one receiver) rather than **collective** (allreduce). First condition with no allreduce anywhere.

## Architecture decisions

### 1. Manual send/recv for PP (not PiPPy)

PiPPy maps one rank per stage. T13 needs 8 ranks (TP group) per stage. Composing PiPPy with DeviceMesh TP is fragile and under-documented. Manual `dist.send`/`dist.recv` gives precise control — only `local_rank=0` on each node does the IB transfer, then broadcasts within the TP group via NVLink. This is how Megatron-LM does it.

### 2. TP via DeviceMesh (same as T11)

2D mesh `(2, 8)` named `("pp", "tp")`, extract `tp_mesh` for `parallelize_module`. Same ColwiseParallel/RowwiseParallel pattern as T11. No `replicate()` call (no DP).

### 3. GPipe with 4 micro-batches

Model split in half: 14 layers per stage. Stage 0 = embedding + layers 0-13, Stage 1 = layers 14-27 + ln_f + head. BATCH_SIZE=16 split into 4 micro-batches of 4. Bubble fraction = (2-1)/(2+4-1) = 20%.

### 4. Activation transfer

After TP's RowwiseParallel, activations are replicated across all 8 GPUs. Only `local_rank=0` sends/recvs over IB. Receiving side broadcasts within TP group via NVLink.

Activation shape: `(4, 2048, 3072)` in bfloat16 = ~48 MiB per micro-batch transfer.

Tags: `mb_idx` for forward, `N_MICROBATCHES + mb_idx` for backward.

### 5. `os._exit(0)` workaround

Same TP cleanup hang as T11. Use `os._exit(0)` after flushing collectors.

## Model structure

```python
class GPTStage0(nn.Module):
    """Embedding + first 14 transformer blocks."""
    # forward(idx) -> hidden states (B, T, D_MODEL)

class GPTStage1(nn.Module):
    """Last 14 transformer blocks + ln_f + output head."""
    # forward(hidden) -> logits (B, T, VOCAB_SIZE)
```

Each node instantiates only its own stage. TP sharding applied to both via same pattern as T11.

## GPipe step pseudocode

```
FORWARD (micro-batches 0..3):
  Stage 0: generate input → model forward → local_rank=0 sends hidden to stage 1
  Stage 1: local_rank=0 recvs hidden → broadcast to TP group → model forward → compute loss

BACKWARD (micro-batches 3..0, reversed):
  Stage 1: loss.backward() → local_rank=0 sends input grad to stage 0
  Stage 0: local_rank=0 recvs grad → broadcast to TP group → hidden.backward(grad)

Both stages: optimizer.step()
```

## Pitfalls to watch

1. **DeviceMesh**: use `(2, 8)` mesh like T11, just name it `("pp", "tp")` instead of `("dp", "tp")`
2. **Autocast scope**: keep `torch.amp.autocast` around model forward only, not around send/recv
3. **Gradient flow**: received activation on stage 1 needs `requires_grad_(True)` before forward
4. **Deadlock ordering**: both stages must process micro-batches in same order (forward 0-3, backward 3-0)
5. **NCCL send/recv tags**: verify tag support works across nodes over IB (fallback: sequential ordering)
6. **Memory**: 4 saved activations × 48 MiB = 192 MiB, negligible on 80GB GPUs

## Config

```python
DURATION_S     = 5 * 60
WARMUP_S       = 30
D_MODEL        = 3072      # same as T1/T11
N_LAYERS       = 28        # 14 per stage
N_HEADS        = 24
FFN_MULT       = 4
SEQ_LEN        = 2048
BATCH_SIZE     = 16        # total, split into micro-batches
N_MICROBATCHES = 4
VOCAB_SIZE     = 32000
LR             = 3e-4
```

## Output files

- `data/t13_node{N}_telemetry.csv` (DCGM, 10Hz)
- `data/t13_node{N}_ib.csv` (InfiniBand, 10Hz)
- `data/t13_node{N}_bmc.csv` (BMC, 2s)

## Launch

```bash
# Node 0:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
  --master_addr=192.168.242.186 --master_port=29500 \
  workloads/train_t13.py

# Node 1:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
  --master_addr=192.168.242.186 --master_port=29500 \
  workloads/train_t13.py
```

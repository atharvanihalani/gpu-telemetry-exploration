# T12 — MoE Training (Expert Parallelism + DP across nodes)

## Context

MoE is now the dominant frontier architecture (DeepSeek-V3, Llama 4, Mixtral, Gemini). It produces a fundamentally different communication pattern from DP or TP: **all-to-all token shuffles** that are data-dependent, asymmetric, and variable per batch. This is a new signal class our classifier hasn't seen.

## Architecture

Mixtral-style: dense attention, sparse MoE FFN. Every other transformer block replaces the dense FFN with a MoE layer (8 experts, top-2 routing).

```
Block 0:  Attention → Dense FFN          (standard)
Block 1:  Attention → MoE(8 experts, top-2)  (sparse)
Block 2:  Attention → Dense FFN
Block 3:  Attention → MoE(8 experts, top-2)
...
```

Each MoE layer:
1. **Router**: `nn.Linear(d_model, n_experts)` → softmax → top-2 selection
2. **Dispatch**: `all_to_all_single` sends tokens to GPU hosting selected expert
3. **Expert compute**: each GPU runs its local expert FFN
4. **Gather**: `all_to_all_single` returns processed tokens to source GPU

## Parallelism layout (2 nodes, 16 GPUs)

```
Node 0: 8 experts across GPUs 0-7  ←─── EP (all-to-all, NVLink)
Node 1: 8 experts across GPUs 0-7  ←─── EP (all-to-all, NVLink)
         │                   │
         └─── DP allreduce ──┘      ←─── DP (IB)
```

- **EP within each node**: 8 experts on 8 GPUs. All-to-all over NVLink.
- **DP across nodes**: gradient allreduce over IB (same as T10).
- Dense attention layers: replicated on all GPUs (like standard DDP).

## Implementation: `workloads/train_t12.py`

~250 lines. Custom MoE layer using `torch.distributed.all_to_all_single`. Reuses T10 multi-node telemetry pattern.

### MoE layer (core ~80 lines)

```python
class MoELayer(nn.Module):
    def __init__(self, d_model, ffn_mult, n_experts, top_k, ep_group):
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, ffn_mult * d_model, bias=False),
                nn.GELU(),
                nn.Linear(ffn_mult * d_model, d_model, bias=False),
            ) for _ in range(n_experts // ep_group.size())  # local experts only
        ])
        self.top_k = top_k
        self.ep_group = ep_group

    def forward(self, x):
        # 1. Route: pick top-k experts per token
        scores = self.router(x)  # (B*T, n_experts)
        weights, indices = torch.topk(scores.softmax(-1), self.top_k)

        # 2. Dispatch: all-to-all sends tokens to expert-hosting GPUs
        # (permutation + all_to_all_single)

        # 3. Expert compute: each GPU runs its local expert(s)

        # 4. Gather: all-to-all returns processed tokens
        # (all_to_all_single + unpermutation + weighted combine)
```

### Model config

Same base as T1 but with MoE blocks:
- d_model=3072, n_layers=28, n_heads=24 (same)
- 14 MoE layers (every other block), 14 dense layers
- 8 experts per MoE layer, top-2 routing
- Each expert = same FFN size as T1 dense FFN (d_model → 4*d_model → d_model)
- Total params: ~3.37B dense + ~3.37B×7 sparse ≈ **27B total, ~6.7B active per token**

### Process groups

Need two separate groups:
- **EP group**: GPUs within each node (for all-to-all)
- **DP group**: corresponding GPUs across nodes (for gradient allreduce)

```python
# EP: [0,1,2,3,4,5,6,7] and [8,9,10,11,12,13,14,15]
# DP: [0,8], [1,9], [2,10], ...
```

Can use `DeviceMesh` with `(dp=2, ep=8)` — same shape as T11's `(dp=2, tp=8)`.

### Telemetry

Same three-collector pattern as T10/T11.

## Expected telemetry vs T10/T11

| Signal | T10 (pure DP) | T11 (TP+DP) | T12 (MoE EP+DP) |
|---|---|---|---|
| NVLink pattern | Periodic bursts | Continuous high | **Variable, data-dependent** |
| NVLink volume | 18 GB/s | 56 GB/s | **Medium** (~20-40 GB/s est.) |
| NVLink autocorr | High (periodic) | Low (continuous) | **Low** (variable per batch) |
| IB traffic | Periodic (DP) | High (TP leaks) | **Periodic (DP)** |
| Power | 687W | 508W | **~500-600W** (sparse compute) |

Key novelty: **variable NVLink traffic per step** — routing decisions differ each batch, so some experts get more tokens than others, producing uneven all-to-all volumes.

## Review points (for Atharva)

- **Expert count**: 8 experts matches our GPU count nicely (1 expert per GPU). Fewer experts (4) would mean 2 per GPU and less all-to-all traffic. More (16+) would require multiple all-to-all rounds. **8 is the natural choice — any objections?**
- **Top-K**: top-2 is the Mixtral/GShard standard. Top-1 (Switch Transformer) would halve communication. **Going with top-2 unless you want to test both.**
- **MoE frequency**: every other block (14/28 MoE layers). Could do every block for maximum all-to-all traffic, or every 4th for minimal. **Every other is the standard.**

## Files to create

- `workloads/train_t12.py` — MoE training with EP + DP
- `plans/t12_moe_ep_dp.md` — copy of this plan

## Verification

1. Training runs at WORLD_SIZE=16
2. All-to-all traffic visible in NVLink (variable per step, not periodic)
3. DP allreduce visible in IB (periodic, same as T10)
4. Router learns non-uniform expert assignment (some experts get more tokens)
5. `nvlink_autocorr_peak` is low (variable ≠ periodic)

## Launch

Same pattern as T10/T11:
```bash
# Node 0:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
  --master_addr=192.168.242.186 --master_port=29500 \
  workloads/train_t12.py

# Node 1: (same with --node_rank=1)
```

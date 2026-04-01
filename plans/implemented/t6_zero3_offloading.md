# T6 — ZeRO-3 / CPU Offloading

## Goal
Shard optimizer states, gradients, and parameters across GPUs + offload to CPU RAM. Tests whether the "high GPU memory" signal can be eliminated while still training.

## Evasion angle
Nearly eliminates the memory signal — GPU memory can drop to inference levels since only the active layer's parameters are on-GPU at any time. PCIe traffic increases dramatically (constant CPU↔GPU transfers). NVLink is still active for parameter gathering across GPUs, but the pattern changes.

## Implementation approach
New script `train_t6.py`. Use DeepSpeed ZeRO Stage 3 with CPU offloading, wrapping the same GPT model from T1.

### Option A: DeepSpeed (most realistic)
```python
import deepspeed

ds_config = {
    "train_micro_batch_size_per_gpu": 4,
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "offload_param": {"device": "cpu", "pin_memory": True},
        "overlap_comm": True,
    },
    "bf16": {"enabled": True},
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 3e-4}
    },
}

model, optimizer, _, _ = deepspeed.initialize(
    model=model, config=ds_config
)
```

### Option B: FSDP with CPU offload (PyTorch native)
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload

model = FSDP(
    model,
    cpu_offload=CPUOffload(offload_params=True),
    use_orig_params=True,
)
```

### Key parameters
- Model: same 3.2B GPT, or attempt larger (6.4B+ since memory is offloaded)
- CPU RAM: should have plenty on RunPod (typically 256+ GB system RAM)
- Everything else: same duration, telemetry collector

## Expected telemetry signature
- **Power**: Lower than T1 — GPU is frequently waiting for CPU↔GPU transfers
- **SM util**: Bursty — high during compute, drops during parameter fetching/offloading
- **Memory**: Very low on GPU (~5–10 GB) — close to inference. Most state lives in CPU RAM.
- **PCIe**: Very high bidirectional traffic (parameter/gradient/optimizer state shuffling)
- **NVLink**: Still active — ZeRO-3 gathers parameters from other GPUs before each forward. Different pattern from DDP allreduce but still present.
- **Key question**: With GPU memory at inference levels and power lower, which signals still distinguish training?

## Hardware notes
- Works on both A100 and H100
- H100's higher PCIe Gen5 bandwidth may reduce the CPU offload bottleneck
- CPU RAM availability varies by RunPod instance — check with `free -h`

## Launch
```bash
# DeepSpeed
deepspeed --num_gpus=8 workloads/train_t6.py --deepspeed_config ds_config.json

# Or FSDP
torchrun --nproc_per_node=8 workloads/train_t6.py
```

## Output
```
data/t6_telemetry.csv
```

## Dependencies
- `pip install deepspeed` (for Option A)
- Or PyTorch >= 2.0 for FSDP (already installed)

## Complexity
Medium. DeepSpeed is a config-driven wrapper. FSDP is native PyTorch but has more edge cases. Either way ~50 lines of changes from T1.

---

## Implementation Notes (2026-03-31)

### Approach chosen: Option B — FSDP (native PyTorch)

Used FSDP with `CPUOffload(offload_params=True)` rather than DeepSpeed. Rationale:
- No extra dependency (`deepspeed` not installed on the pod)
- PyTorch 2.4.1 FSDP is mature and stable
- FSDP2 (`fully_shard` composable API) is also available in this PyTorch version, but FSDP1 was chosen for more explicit control over CPU offload behavior and better documentation

### Environment
- PyTorch 2.4.1+cu124
- FSDP + CPUOffload confirmed available
- CPU RAM: ~2015 GB (ample for offloading 3.2B params + AdamW states, which need ~50GB CPU-side)

### Key FSDP configuration
```python
FSDP(
    model,
    auto_wrap_policy=transformer_auto_wrap_policy({TransformerBlock}),
    cpu_offload=CPUOffload(offload_params=True),
    mixed_precision=MixedPrecision(param_dtype=bf16, reduce_dtype=bf16, buffer_dtype=bf16),
    sharding_strategy=ShardingStrategy.FULL_SHARD,   # ZeRO-3 equivalent
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # prefetch next layer during backward
    device_id=local_rank,
    sync_module_states=True,   # broadcast rank-0 weights on init
    use_orig_params=False,
)
```

### Design decisions
1. **Model built on CPU**: FSDP with CPU offload expects the model to start on CPU. It shards and initializes from there.
2. **Each TransformerBlock is an FSDP unit**: The `transformer_auto_wrap_policy` wraps each block individually, so only one block's parameters are materialized on GPU at any time during forward/backward.
3. **No `torch.amp.autocast`**: FSDP's `MixedPrecision` policy handles dtype casting. Using both would conflict.
4. **Fallback**: If CPU offload fails at init, the script falls back to GPU-only FSDP (still FULL_SHARD / ZeRO-3 sharding, but params stay on GPU). This is logged so we know which mode ran.
5. **Logging every 5 steps** (vs T1's every 10): T6 steps are slower due to CPU↔GPU transfer overhead, so fewer steps total. More frequent logging keeps the output useful.

### Expected differences from T1
| Aspect | T1 (DDP) | T6 (FSDP + CPU offload) |
|---|---|---|
| GPU memory | ~67 GB (full replica) | ~5-15 GB (one layer at a time) |
| Power | ~400W sustained | Lower, more variable |
| SM util | ~100% sustained | Bursty (compute between PCIe stalls) |
| PCIe traffic | Low (data loading only) | Very high (constant param streaming) |
| NVLink | DDP allreduce heartbeat | FSDP gather/scatter (different pattern) |
| Steps/sec | ~3-4/min | Much slower (PCIe bottleneck) |

### What to watch for
- GPU memory should drop dramatically — this is the whole point. If it's still >30GB, something is wrong with the offload config.
- PCIe TX+RX should be much higher than T1.
- NVLink traffic should still be present (FSDP gathers parameter shards across GPUs) but the pattern differs from DDP's synchronized allreduce.

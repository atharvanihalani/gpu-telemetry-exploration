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

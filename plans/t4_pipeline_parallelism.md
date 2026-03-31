# T4 — Pipeline Parallelism

## Goal
Split model layers across GPUs (pipeline parallelism) instead of replicating the full model (data parallelism). Produces a fundamentally different inter-GPU communication pattern: sequential activation passing (GPU0→1→2→...→7) instead of all-to-all allreduce.

## Evasion angle
No allreduce at all — the strongest training signal disappears entirely. NVLink traffic becomes peer-to-peer between adjacent pipeline stages. Power and SM util follow a "wave" pattern (only one or a few stages active at a time during the pipeline bubble), not the sustained flat profile of DDP.

## Implementation approach
New script `train_t4.py`. Use PyTorch's `torch.distributed.pipelining` (PiPPy) or manual pipeline with `torch.distributed.rpc`.

### Approach A: Manual pipeline (simpler, more control)
- Split the GPT model into 8 stages (3–4 transformer blocks per GPU)
- GPU 0: embedding + blocks 0–3
- GPU 1: blocks 4–7
- ...
- GPU 7: blocks 24–27 + LN + head
- Forward: send activations GPU_i → GPU_{i+1} via `torch.distributed.send/recv`
- Backward: send gradients in reverse
- Each GPU runs its own optimizer on its local parameters

### Approach B: PiPPy (less boilerplate)
```python
from torch.distributed.pipelining import SplitPoint, pipeline, ScheduleGPipe

# Split model at layer boundaries
split_spec = {f"blocks.{i}": SplitPoint.BEGINNING for i in [4, 8, 12, 16, 20, 24, 27]}
pipe = pipeline(model, mb_args=(microbatch,), split_spec=split_spec)
schedule = ScheduleGPipe(pipe, n_microbatches=8)
schedule.step(x)
```

### Key parameters
- Model: same 3.2B GPT as T1, split across 8 GPUs (~420M params per stage)
- Microbatches: 8 (fills the pipeline to reduce bubble overhead)
- `SEQ_LEN`: 2048, `BATCH_SIZE`: 4
- Duration: 5 min

## Expected telemetry signature
- **Power**: Uneven across GPUs — first/last stages may be busier. "Wave" pattern as microbatches flow through.
- **SM util**: Not flat 100% — each GPU is idle during its pipeline bubble. Expect 50–70% average.
- **Memory**: Lower per GPU (~8–10 GB per stage) since each holds only a fraction of the model
- **NVLink**: Point-to-point traffic between adjacent stages only (GPU0↔1, 1↔2, etc.), not all-to-all. Much lower total NVLink bandwidth than DDP.
- **Key question**: Can a detector distinguish pipeline P2P traffic from inference P2P traffic (tensor parallelism)?

## Hardware notes
- Works on both A100 and H100
- Pipeline bubble fraction is the same regardless of GPU speed
- On H100, each stage completes faster → higher throughput, same bubble ratio

## Launch
```bash
torchrun --nproc_per_node=8 workloads/train_t4.py
```

## Output
```
data/t4_telemetry.csv
```

## Dependencies
- PyTorch >= 2.3 for `torch.distributed.pipelining`
- Or manual send/recv (works with any PyTorch version)

## Complexity
Medium-high. Pipeline parallelism requires careful model splitting and microbatch scheduling. Estimate ~200 lines of new code if doing it manually. PiPPy reduces this but may have compatibility issues.

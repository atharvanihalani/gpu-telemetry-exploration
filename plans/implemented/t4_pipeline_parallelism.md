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

---

## Implementation Notes (2026-03-31)

### Approach used: PiPPy (Approach B)

PyTorch 2.4.1 ships `torch.distributed.pipelining` with full support for `pipeline()`, `SplitPoint`, `ScheduleGPipe`, and `build_stage()`. Verified all APIs are importable and functional.

### Key API details (PyTorch 2.4.1)

```python
# Trace and split model
pipe = pipeline(module=model_cpu, mb_args=(example_microbatch,), split_spec=SPLIT_SPEC)

# Build this rank's stage (moves submodule to device, determines P2P buffer shapes)
stage = pipe.build_stage(rank, device=device)

# Create schedule with loss function (loss only evaluated on last stage)
schedule = ScheduleGPipe(stage=stage, n_microbatches=N, loss_fn=loss_fn)

# Training step:
#   rank 0:     schedule.step(input, target=target, losses=losses)
#   last rank:  schedule.step(target=target, losses=losses)
#   middle:     schedule.step()
```

`pipeline()` uses `torch.export` tracing (not `torch.fx.symbolic_trace`). This handles `torch.arange()` in the forward pass correctly, which `fx.symbolic_trace` cannot.

`ScheduleGPipe.step()` accepts *whole-batch* input and chunks it into microbatches automatically via `torch.tensor_split`. The `mb_args` to `pipeline()` must match the microbatch shape (BATCH_SIZE // N_MICROBATCHES).

### Design decisions

1. **Batch size = 8** (not 4 like T1 per-GPU). With N_MICROBATCHES=8, each microbatch = 1 sample. This fills the pipeline adequately. Total batch is smaller than T1's effective batch (T1: 4 per GPU * 8 GPUs = 32), but pipeline parallelism has inherent bubble overhead that limits throughput regardless.

2. **No explicit bf16 conversion**. Using `torch.amp.autocast` around `schedule.step()` instead. The PiPPy stage initialization runs a forward pass to determine P2P buffer shapes, which is simpler with fp32 parameters. Autocast gives the same compute pattern (tensor core utilization, power profile) for telemetry purposes.

3. **Per-stage optimizer**. Each rank creates an AdamW optimizer over only its local stage parameters. This is the standard pattern for pipeline parallelism — no cross-rank parameter sync needed (unlike DDP's allreduce).

4. **Synthetic random targets on last rank**. Loss is meaningless with random data anyway. The last rank generates its own random targets rather than receiving rank 0's targets (which would require extra communication).

### Stage split (28 layers, 8 stages)

| Stage | GPU | Layers | Notes |
|-------|-----|--------|-------|
| 0 | GPU 0 | embedding + blocks 0-2 | 3 blocks + embedding (~540M params) |
| 1 | GPU 1 | blocks 3-6 | 4 blocks (~480M params) |
| 2 | GPU 2 | blocks 7-9 | 3 blocks (~360M params) |
| 3 | GPU 3 | blocks 10-13 | 4 blocks (~480M params) |
| 4 | GPU 4 | blocks 14-16 | 3 blocks (~360M params) |
| 5 | GPU 5 | blocks 17-20 | 4 blocks (~480M params) |
| 6 | GPU 6 | blocks 21-23 | 3 blocks (~360M params) |
| 7 | GPU 7 | blocks 24-27 + ln_f + head | 4 blocks + head (~580M params) |

Not perfectly balanced — stages 0 and 7 are heavier due to embedding and output head. In a production setup you'd balance by FLOPs, but for telemetry the imbalance adds realism (real pipelines are never perfectly balanced).

### Potential issues to watch for when running

1. **Memory**: Each stage holds ~360-580M params in fp32 (~1.4-2.3GB) plus AdamW states (2x = ~2.8-4.6GB) plus activations for 8 microbatches. Should fit comfortably in 80GB.

2. **Pipeline bubble**: GPipe has a bubble fraction of (num_stages - 1) / (num_stages + num_microbatches - 1) = 7/15 = 47%. Expect ~50% average SM utilization, which is a key distinguishing signal vs T1's ~100%.

3. **NVLink traffic pattern**: Expect traffic only on links between adjacent GPU pairs (0-1, 1-2, ..., 6-7), with the NVSwitch routing P2P. The all-to-all pattern from DDP allreduce should be completely absent.

4. **Step time**: Each step processes 8 samples (vs T1's 32). With 47% bubble overhead, expect lower throughput. Step time depends on the slowest stage.

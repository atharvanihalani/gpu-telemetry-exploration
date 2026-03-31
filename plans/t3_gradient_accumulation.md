# T3 — Gradient Accumulation

## Goal
Accumulate N microbatch gradients before running allreduce. This stretches the allreduce heartbeat period by N×, testing whether a detector keyed on "allreduce every ~1s" can be fooled by reducing communication frequency.

## Evasion angle
A detector tuned to T1's heartbeat period (~15–20s per step on A100) would miss T3 if accumulation steps are large enough to push the period outside the expected window. Power and SM util remain high throughout — only the NVLink burst frequency changes.

## Implementation approach
Fork `train_t1.py` → `train_t3.py`. Add gradient accumulation loop: run N forward+backward passes (with `DDP.no_sync()` context manager to suppress allreduce), then sync on the Nth step.

### Key code change
```python
ACCUM_STEPS = 8  # tune this — higher = less frequent allreduce

for micro_step in range(ACCUM_STEPS):
    x, y = get_batch()

    # Suppress allreduce on all but the last microstep
    ctx = model.no_sync() if micro_step < ACCUM_STEPS - 1 else nullcontext()
    with ctx:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            loss = loss / ACCUM_STEPS  # scale loss
        loss.backward()

optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

### Key parameters
- `ACCUM_STEPS`: 8 (primary variable — try 4, 8, 16 to map detection threshold)
- Model config: same as T1 (3.2B params)
- Everything else: same as T1

## Expected telemetry signature
- **Power**: Same sustained high as T1 (~400W A100 / ~700W H100) — GPU is always computing
- **SM util**: ~100% (same as T1)
- **Memory**: Same as T1 (gradients accumulate in-place)
- **NVLink**: Bursts spaced ACCUM_STEPS × further apart. With `ACCUM_STEPS=8`, heartbeat period goes from ~15–20s to ~2–3 min
- **Key question**: At what accumulation factor does the heartbeat become undetectable at 1Hz sampling?

## Hardware notes
- Works identically on A100 and H100
- On H100 (faster steps), each accumulation window is shorter in wall-clock time, so may need higher ACCUM_STEPS to achieve the same evasion effect

## Launch
```bash
torchrun --nproc_per_node=8 workloads/train_t3.py
```

## Output
```
data/t3_telemetry.csv
```

## Dependencies
- Same as T1: PyTorch, pynvml
- `from contextlib import nullcontext`

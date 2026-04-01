# E3 — Intermittent Training (Duty-Cycled)

## Goal
Train for 30s, sleep 10s, repeat. Breaks the "sustained flat power" assumption that distinguishes training from inference.

## Evasion angle
A detector looking for "sustained high power over minutes" would see this as bursty — similar to inference or periodic batch jobs. Tests whether temporal pattern *over minutes* is required for detection, vs. per-second features.

## Implementation approach
Fork `train_t1.py` → `train_e3.py`. Add a duty cycle wrapper around the training loop.

### Key code
```python
TRAIN_WINDOW_S = 30   # train for this long
SLEEP_WINDOW_S = 10   # then idle for this long
# Duty cycle = 30/40 = 75%

t_start = time.time()
while time.time() - t_start < DURATION_S:
    # Training burst
    burst_start = time.time()
    while time.time() - burst_start < TRAIN_WINDOW_S:
        x, y = get_batch()
        # ... normal training step ...

    # Sleep (GPU goes idle)
    if collector and is_rank0:
        collector.set_phase("sleep")
    dist.barrier()  # sync all GPUs before sleeping
    time.sleep(SLEEP_WINDOW_S)
    if collector and is_rank0:
        collector.set_phase("steady")
    dist.barrier()
```

### Key parameters
- `TRAIN_WINDOW_S`: 30 (primary variable — try 10, 30, 60)
- `SLEEP_WINDOW_S`: 10 (try 5, 10, 30)
- Duty cycle: 75% at default settings
- Model: same 3.2B GPT as T1
- Duration: 5 min total

## Expected telemetry signature
- **Power**: Square wave — ~400W for 30s, then drops to ~60–80W for 10s, repeats
- **SM util**: Same square wave — 100% during training bursts, 0% during sleep
- **Memory**: Stays high throughout (model + optimizer states remain allocated during sleep)
- **NVLink**: Allreduce heartbeat during training bursts, zero during sleep
- **Temperature**: Sawtooth — rises during bursts, falls during sleep
- **Key question**: Is the "memory stays high during sleep" signal detectable? A model loaded for inference also has high memory but zero SM util.

## Variants to try
| Config | Train | Sleep | Duty cycle | Evasion strength |
|---|---|---|---|---|
| Default | 30s | 10s | 75% | Low — still looks mostly like training |
| Aggressive | 10s | 30s | 25% | High — mostly idle, short training bursts |
| Matched | 10s | 10s | 50% | Medium — equal on/off |

## Hardware notes
- Works on both A100 and H100 with no changes
- `dist.barrier()` ensures all GPUs enter/exit sleep simultaneously
- On H100, more training steps fit in each burst (faster compute)

## Launch
```bash
torchrun --nproc_per_node=8 workloads/train_e3.py
```

## Output
```
data/e3_telemetry.csv
```

## Dependencies
- Same as T1

## Complexity
Low. Small modification to T1's training loop.

---

## Implementation notes (2026-03-31)

### What was built

`workloads/train_e3.py` — fork of `train_t1.py` with duty cycle wrapper.

### Structure

The training loop is split into two phases:

1. **Warmup phase** (0 to `WARMUP_S`): Continuous training, identical to T1. Phase label = "warmup". This lets the model, optimizer states, and GPU thermals reach steady state before duty cycling begins.

2. **Duty cycle phase** (`WARMUP_S` to `DURATION_S`): Alternates between training bursts (`TRAIN_WINDOW_S`) and sleep windows (`SLEEP_WINDOW_S`). Phase labels alternate between "steady" (training) and "sleep" (idle).

### Synchronization

- `dist.barrier()` called **before** entering sleep (ensures all GPUs finish their last training step before any GPU goes idle).
- `dist.barrier()` called **after** sleep (ensures all GPUs wake up together before resuming training).
- This prevents DDP deadlocks where one GPU starts a forward pass (triggering allreduce hooks) while another is still sleeping.

### Telemetry labels

The phase column in `data/e3_telemetry.csv` cycles through:
- `warmup` -> `steady` -> `sleep` -> `steady` -> `sleep` -> ... -> `cooldown`

This allows analysis scripts to filter by phase and compute per-burst vs per-sleep statistics.

### Key differences from T1

| Aspect | T1 | E3 |
|---|---|---|
| Training loop | Single continuous loop | Two-phase: warmup then duty-cycled |
| Phase labels | warmup, steady, cooldown | warmup, steady, sleep (alternating), cooldown |
| Sleep periods | None | `SLEEP_WINDOW_S` idle gaps between bursts |
| Barriers | None (DDP handles sync internally) | Explicit `dist.barrier()` around sleep windows |
| Output CSV | `data/t1_telemetry.csv` | `data/e3_telemetry.csv` |

### Expected duty cycles at default settings

With `WARMUP_S=30`, `TRAIN_WINDOW_S=30`, `SLEEP_WINDOW_S=10`, `DURATION_S=300`:

- 0-30s: warmup (continuous training)
- 30-60s: training burst 1
- 60-70s: sleep 1
- 70-100s: training burst 2
- 100-110s: sleep 2
- ... repeating ...
- ~290-300s: final burst (may be truncated)
- 300-305s: cooldown

Approximately 6-7 duty cycles after warmup.

### Not yet run

Script created and verified to parse (`python -c "import workloads.train_e3"` succeeds). Actual execution with `torchrun --nproc_per_node=8 workloads/train_e3.py` deferred to next GPU session.

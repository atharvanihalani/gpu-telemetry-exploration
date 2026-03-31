# E5 — Smoothed Allreduce (Large NCCL Buffer)

## Goal
Stretch the allreduce communication window by increasing NCCL buffer sizes and forcing the ring algorithm. Turns the sharp NVLink burst into a low, continuous rate. Tests whether burst *shape* is load-bearing in detection.

## Evasion angle
A detector looking for sharp NVLink spikes (the allreduce "heartbeat") might miss a smoothed-out version where NVLink traffic is spread over a longer window at lower peak bandwidth. The total bytes transferred are the same, but the temporal profile changes.

## Implementation approach
No code changes — NCCL tuning via environment variables.

### Environment variables
```bash
# Force ring algorithm (predictable, stretchable)
NCCL_ALGO=Ring

# Large buffer = longer transfer window, lower peak rate
NCCL_BUFFSIZE=134217728  # 128MB (default is 4MB)

# Optionally limit bandwidth to spread transfer further
NCCL_MAX_NCHANNELS=1     # Use only 1 channel (default: auto, typically 8-12)
```

### Launch command
```bash
NCCL_ALGO=Ring \
NCCL_BUFFSIZE=134217728 \
NCCL_MAX_NCHANNELS=1 \
torchrun --nproc_per_node=8 workloads/train_t1.py
```

### Key parameters
- Same model and config as T1
- The only changes are NCCL environment variables
- Try multiple configs to map the "smoothing" space:

| Config | NCCL_BUFFSIZE | NCCL_MAX_NCHANNELS | Expected effect |
|---|---|---|---|
| Mild | 16MB | 4 | Slightly broader peaks |
| Medium | 128MB | 2 | Notably smoothed |
| Aggressive | 128MB | 1 | Maximum smoothing |

## Expected telemetry signature
- **Power**: Similar average to T1, but the periodic dips may be shallower (allreduce overlaps more with compute)
- **SM util**: Similar to T1 — compute is unchanged
- **Memory**: Same as T1
- **NVLink**: Same total bytes, but spread over longer window. Peak bandwidth lower, duration longer. The "sharp spike" becomes a "broad bump."
- **Key question**: At what smoothing level does the heartbeat become undetectable at 1Hz sampling? If the allreduce window stretches to >1s, it may look like continuous background traffic.

## Hardware notes
- Works on both A100 and H100
- H100's higher NVLink bandwidth means more smoothing headroom before becoming a bottleneck
- `NCCL_MAX_NCHANNELS=1` will significantly slow allreduce — training throughput drops
- Verify actual NVLink rates with DCGM during the run

## Launch
```bash
NCCL_ALGO=Ring NCCL_BUFFSIZE=134217728 NCCL_MAX_NCHANNELS=1 \
torchrun --nproc_per_node=8 workloads/train_t1.py
```

## Output
```
data/e5_telemetry.csv
```

## Dependencies
- Same as T1 (no additional packages)

## Complexity
Low. Environment variables only, no code changes. But interpreting the results requires DCGM per-link NVLink data at sub-second resolution to see the smoothing effect.

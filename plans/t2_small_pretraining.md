# T2 — Small Pre-Training

## Goal
Same DDP training pattern as T1 but with a significantly smaller model. Tests whether **temporal pattern** (allreduce heartbeat, cross-GPU synchronization) is a better discriminator than **magnitude** (raw power, memory usage).

## Implementation approach
Fork `train_t1.py` → `train_t2.py`. Only change: smaller model config. Everything else identical (DDP, synthetic data, same duration, same telemetry collector).

### Model config
| Param | T1 (large) | T2 (small) |
|---|---|---|
| d_model | 3072 | 768 |
| n_layers | 28 | 12 |
| n_heads | 24 | 12 |
| ffn_mult | 4 | 4 |
| Approx params | ~3.2B | ~125M |
| Est. GPU mem | ~38 GB | ~1.5 GB |

The 125M config is deliberately small — it should use <5% of VRAM, making memory usage close to inference levels. Power and SM util will still be high during compute, but with much shorter step times.

### Key parameters
- `BATCH_SIZE`: Start at 4 (same as T1). Could increase since memory is plentiful — but keeping it the same isolates the model-size variable.
- `SEQ_LEN`: 2048 (same as T1)
- `DURATION_S`: 300 (5 min, same as T1)
- `WARMUP_S`: 30

## Expected telemetry signature
- **Power**: Lower than T1 (~150–250W on A100, ~200–400W on H100) — GPU not fully saturated
- **SM util**: Still high during forward/backward, but steps are much faster → more frequent idle gaps between steps
- **Memory**: Very low (~2–3 GB) — close to inference levels
- **NVLink**: Same allreduce heartbeat as T1 but at higher frequency (shorter steps = more frequent allreduce)
- **Key question**: Does the heartbeat pattern persist even when power/memory look inference-like?

## Hardware notes
- Works on both A100 and H100 with no changes (model fits trivially on 80GB)
- Steps will be significantly faster on H100 — allreduce frequency will be higher

## Launch
```bash
torchrun --nproc_per_node=8 workloads/train_t2.py
```

## Output
```
data/t2_telemetry.csv
```

## Dependencies
- Same as T1: PyTorch, pynvml

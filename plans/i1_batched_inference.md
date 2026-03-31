# I1 — Standard Batched Inference

## Goal
Fixed-size batch inference (non-autoregressive). The "standard negative" baseline — a step up from I2's single-stream autoregressive generation, but still clearly not training.

## Implementation approach
New script `infer_i1.py`. Load Llama-3.1-8B on each GPU (same as I2), but run fixed-size batches through the model's forward pass only (no generation loop). This exercises the prefill path, not the decode path.

### Key code
```python
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    while time.time() - t_start < duration_s:
        input_ids = torch.randint(100, vocab_size - 100, (BATCH_SIZE, SEQ_LEN), device=device)
        # Just forward pass — no autoregressive generation
        outputs = model(input_ids)
        # Discard outputs; we only care about telemetry
```

### Key parameters
- Model: Llama-3.1-8B (same as I2)
- `BATCH_SIZE`: 8–16 (higher than I2 to stress compute)
- `SEQ_LEN`: 2048
- Setup: 8 independent GPU streams (same as I2, no tensor parallelism)
- Duration: 5 min

## Expected telemetry signature
- **Power**: Moderate (~150–250W on A100, ~200–400W on H100) — higher than I2 because batching improves compute utilization
- **SM util**: Moderate-high (~40–70%) — batched prefill is more compute-bound than autoregressive decode
- **Memory**: Same as I2 (~17 GB per GPU) — no optimizer states, just weights + activations
- **NVLink**: Near zero (no inter-GPU communication)
- **PCIe**: Low (no data loading beyond initial model)
- **Key question**: How does batched inference SM util compare to T2 (small training)? Both could be in a similar range.

## Hardware notes
- Works on both A100 and H100 with no changes
- Batch size may need tuning — start at 8, increase if memory allows

## Launch
```bash
python workloads/infer_i1.py
```

## Output
```
data/i1_telemetry.csv
```

## Dependencies
- Same as I2: transformers, accelerate, pynvml

# B3 — CUDA Graph Warmup / torch.compile

## Goal
Capture the telemetry signature of JIT compilation and CUDA graph capture — one-time overhead at the start of a workload. Important to know so it doesn't get misclassified as a sustained compute event.

## Implementation approach
New script `baseline_b3.py`. Use `torch.compile` on the GPT model and run a few warmup iterations that trigger compilation, then sit idle.

### Key code
```python
model = GPT(...).to(device).to(torch.bfloat16)
compiled_model = torch.compile(model)

# Trigger compilation with a few forward passes
for _ in range(5):
    x = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), device=device)
    with torch.no_grad():
        _ = compiled_model(x)

# Sit idle for the rest of the duration to show the contrast
time.sleep(DURATION_S - warmup_elapsed)
```

### Key parameters
- Model: 3.2B GPT (large enough to have non-trivial compile time)
- Compilation: `torch.compile(mode="default")` — takes ~30–60s on this model
- Duration: 5 min total (compile phase + idle)

## Expected telemetry signature
- **During compilation (~30–60s)**: Moderate SM util (compiler kernels running), moderate power, brief bursts. Looks superficially like inference but is not sustained.
- **After compilation**: Drops to idle baseline (same as B1 if model is loaded, or empty idle)
- **Memory**: Model weights + compilation cache (~20–30 GB during compile, drops after)
- **NVLink**: Zero
- **Key finding**: Short transient, not sustained. A detector with a minimum duration threshold (e.g., "high SM util for >2 min") would correctly ignore this.

## Launch
```bash
python workloads/baseline_b3.py
```

## Output
```
data/b3_telemetry.csv
```

## Complexity
Low — ~40 lines. The interesting part is the telemetry, not the code.

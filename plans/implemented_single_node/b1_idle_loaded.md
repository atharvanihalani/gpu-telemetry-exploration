# B1 — Idle with Model Loaded

## Goal
Load a large model into GPU memory, then sit idle. Establishes a baseline for "has a model" vs. "is running a model" — important to avoid false positives on idle inference servers.

## Implementation approach
New script `baseline_b1.py`. Load Llama-3.1-8B (same as I2) on each GPU, then sleep for the duration.

### Key code
```python
# Load model on each GPU (same as I2)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": device}
)
model.eval()

# Just sit idle
time.sleep(DURATION_S)
```

### Key parameters
- Model: Llama-3.1-8B (~16 GB per GPU)
- Duration: 5 min
- Setup: 8 GPUs, one model each

## Expected telemetry signature
- **Power**: Idle (~60–80W) — same as empty idle
- **SM util**: 0%
- **Memory**: ~17 GB per GPU (model weights loaded)
- **NVLink**: Zero
- **Temperature**: Low, stable
- **Key finding**: High memory + zero compute. If a detector uses memory threshold alone, this is a false positive. If it requires memory + SM util, this is correctly classified as idle.

## Launch
```bash
python workloads/baseline_b1.py
```

## Output
```
data/b1_telemetry.csv
```

## Complexity
Trivial — ~30 lines.

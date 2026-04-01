# B2 — Checkpoint Save / Load

## Goal
Serialize a large model checkpoint to disk, then load it back. Captures the telemetry signature of a short-lived I/O spike that shouldn't be mistaken for sustained training or inference.

## Implementation approach
New script `baseline_b2.py`. Build the 3.2B GPT model (same as T1), save it, load it, repeat for the duration.

### Key code
```python
# Build model (same as T1)
model = GPT(...).to(device).to(torch.bfloat16)

# Repeated save/load cycle
while time.time() - t_start < DURATION_S:
    # Save
    torch.save(model.state_dict(), f"/tmp/checkpoint_gpu{gpu_id}.pt")
    # Load
    state_dict = torch.load(f"/tmp/checkpoint_gpu{gpu_id}.pt", map_location=device)
    model.load_state_dict(state_dict)
```

### Key parameters
- Model: 3.2B GPT (same as T1) — large enough for visible I/O spikes
- Duration: 5 min (repeated save/load cycles)
- Storage: `/tmp` (local NVMe on RunPod)

## Expected telemetry signature
- **Power**: Brief spikes during load (GPU memory writes), mostly idle between operations
- **SM util**: Near zero (save/load is I/O, not compute)
- **Memory**: Spikes during load (model weights + temporary buffers), drops between cycles
- **PCIe**: High during save/load (disk → CPU → GPU path)
- **NVLink**: Zero (no inter-GPU communication)
- **Key finding**: PCIe spikes without SM util or NVLink — distinguishable from both training and inference

## Launch
```bash
python workloads/baseline_b2.py
```

## Output
```
data/b2_telemetry.csv
```

## Complexity
Low — ~50 lines.

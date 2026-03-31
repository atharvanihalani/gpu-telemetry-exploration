# E2 — Fake Inference Cover Traffic

## Goal
Run inference on GPUs 0–3 simultaneously with training on GPUs 4–7. Tests whether splitting a node between training and inference can confuse aggregate-level detectors.

## Evasion angle
If a detector only looks at node-level aggregates (total power, average SM util), the inference workload dilutes the training signal. A per-GPU detector should still catch it — this experiment tests where the boundary is.

## Implementation approach
New script `run_e2.py` that orchestrates two sub-processes:

### Architecture
```
GPUs 0–3: Inference (I2-style, streaming autoregressive)
GPUs 4–7: Training (T1-style, DDP across 4 GPUs only)
```

### Key code
```python
import subprocess, os

# Launch training on GPUs 4-7
train_env = os.environ.copy()
train_env["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
train_proc = subprocess.Popen(
    ["torchrun", "--nproc_per_node=4", "workloads/train_t1.py"],
    env=train_env
)

# Launch inference on GPUs 0-3
infer_env = os.environ.copy()
infer_env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
infer_proc = subprocess.Popen(
    ["python", "workloads/infer_i2.py"],  # needs modification to use 4 GPUs
    env=infer_env
)
```

### Modifications needed
- `train_t1.py`: needs to work with `--nproc_per_node=4` (already should — DDP is world-size agnostic)
- `infer_i2.py`: needs to respect `CUDA_VISIBLE_DEVICES` for GPU count (already uses `torch.cuda.device_count()`)
- Telemetry collector: needs to collect from ALL 8 GPUs regardless of which process runs on which — run a single collector from the orchestrator

### Key parameters
- Training GPUs: 4–7 (4-GPU DDP)
- Inference GPUs: 0–3 (4 independent streams)
- Duration: 5 min
- Model configs: same as T1/I2

## Expected telemetry signature
- **Per-GPU power**: GPUs 0–3 at ~88W (inference), GPUs 4–7 at ~400W (training) — bimodal
- **Aggregate power**: Average ~244W — neither clearly training nor inference
- **SM util**: GPUs 0–3 at ~5%, GPUs 4–7 at ~100% — bimodal
- **NVLink**: GPUs 4–7 show allreduce heartbeat (4-GPU DDP); GPUs 0–3 show nothing
- **Memory**: GPUs 4–7 at ~66 GB; GPUs 0–3 at ~17 GB
- **Key question**: Does a per-GPU detector catch the training GPUs? Does an aggregate-only detector miss it?

## Hardware notes
- Works on both A100 and H100
- NVSwitch topology means GPUs 4–7 can still do all-to-all allreduce (they're all connected)
- Need to verify that `CUDA_VISIBLE_DEVICES` correctly isolates the two workloads

## Launch
```bash
python workloads/run_e2.py
```

## Output
```
data/e2_telemetry.csv
```

## Dependencies
- Same as T1 + I2

## Complexity
Medium. Orchestrating two concurrent workloads with different GPU assignments. Main risk: telemetry collector needs to see all 8 GPUs from a single process.

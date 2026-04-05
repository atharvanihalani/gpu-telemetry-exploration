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

---

## Implementation notes (2026-03-31)

### What was implemented

1. **`workloads/run_e2.py`** — orchestrator script that:
   - Starts a single `TelemetryCollector` on all 8 GPUs → `data/e2_telemetry.csv`
   - Launches training on GPUs 4–7 via `torchrun --nproc_per_node=4 --master_port=29501`
   - Launches inference on GPUs 0–3 via `python workloads/infer_i2.py`
   - Both sub-processes get `TELEMETRY_DISABLED=1` so only the orchestrator collects
   - Phase labels: loading → warmup (30s) → steady → cooldown (5s)
   - Monitors both processes; if one dies, kills the other and stops cleanly
   - Uses `master_port=29501` to avoid conflicts with default 29500

2. **`workloads/train_t1.py`** — already had `TELEMETRY_DISABLED` support and guarded collector calls (no changes needed)

3. **`workloads/infer_i2.py`** — added guards (`if collector:`) around all `collector.set_phase()`, `collector.stop()` calls that were previously unguarded (lines 148–160 in the original)

### Design decisions

- **10s init wait**: The orchestrator waits 10s after launching both sub-processes before entering "warmup". This gives DDP init and model loading time to start, but the real warmup happens during the 30s warmup phase.
- **Process health checks**: Polled every 1s during warmup and steady phases. If either process exits, the other is killed immediately.
- **Graceful shutdown**: SIGTERM first, then SIGKILL after 10s timeout.
- **No `__init__.py`**: The workloads directory doesn't need one — scripts use `sys.path.insert` for imports, consistent with existing T1/I2 scripts.

### Important: CUDA_VISIBLE_DEVICES remapping

When `CUDA_VISIBLE_DEVICES=4,5,6,7` is set, the training sub-process sees those as `cuda:0` through `cuda:3` internally. The telemetry collector in the orchestrator (which runs without `CUDA_VISIBLE_DEVICES` restriction) sees the actual GPU indices 0–7. This means:
- Training telemetry will show activity on physical GPUs 4–7
- Inference telemetry will show activity on physical GPUs 0–3
- The CSV correctly records physical GPU indices since the collector runs in the orchestrator process

### Verification

All three scripts pass syntax check (`ast.parse`) and module import (`import workloads.run_e2` etc.).

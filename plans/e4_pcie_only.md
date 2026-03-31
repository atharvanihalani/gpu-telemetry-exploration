# E4 — PCIe-Only Allreduce

## Goal
Force NCCL to use PCIe instead of NVLink for allreduce. The NVLink heartbeat — the strongest training signal — goes completely dark. Tests whether other signals (PCIe pattern, SM util, memory, power) can compensate.

## Evasion angle
NVLink traffic is the "smoking gun" for training. If an adversary disables NVLink for NCCL (at the cost of much slower training), NVLink-based detectors see nothing. Allreduce traffic shifts to PCIe, which is harder to distinguish from data loading or inference I/O.

## Implementation

Dedicated script: `workloads/train_e4.py` — a thin wrapper around T1 that imports the model class and config constants from `train_t1.py`, sets the NCCL env vars before any NCCL init, and overrides duration/output path.

### Launch command
```bash
torchrun --nproc_per_node=8 workloads/train_e4.py
```

### What the script does differently from T1
- Sets `NCCL_P2P_DISABLE=1` and `NCCL_SHM_DISABLE=1` at the top of the file (before `torch.distributed` is imported), forcing NCCL to use PCIe instead of NVLink/NVSwitch
- `DURATION_S = 10 * 60` (10 min, doubled from T1's 5 min — PCIe allreduce is much slower)
- `OUTPUT_CSV = data/e4_telemetry.csv` (overridable via `E4_OUTPUT_CSV` env var)
- Logs the NCCL env var values at startup for confirmation
- Imports `GPT`, all config constants, and `WARMUP_S` from `train_t1.py` — no model code duplication

### Verify NVLink is actually dark during the run
```bash
nvidia-smi dmon -s n -d 1
# Should show near-zero NVLink TX/RX
```

### Key parameters
- Model: same 3.37B GPT as T1 (imported, no changes)
- The ONLY difference from T1 is the NCCL transport path and longer duration
- Training will be significantly slower (~5–10× due to PCIe bandwidth bottleneck)

## Expected telemetry signature
- **Power**: Similar to T1 but potentially lower (GPU spends more time waiting for allreduce over slow PCIe)
- **SM util**: Lower than T1 — longer communication phases mean more idle time between compute
- **Memory**: Same as T1 (~66 GB)
- **NVLink**: **Zero** — this is the point. Completely dark.
- **PCIe**: Very high — allreduce traffic now flows over PCIe. Should show periodic bursts matching the allreduce heartbeat, but on PCIe instead of NVLink.
- **Key question**: Does the PCIe heartbeat pattern replicate the NVLink heartbeat? Is PCIe-level periodicity detectable?

## Hardware notes
- Works on both A100 and H100
- A100 PCIe Gen4 x16: ~32 GB/s → allreduce on 3.2B params (~12 GB of gradients) takes ~0.4s per step
- H100 PCIe Gen5 x16: ~64 GB/s → ~0.2s per step
- Training throughput drops dramatically — a 5-min run may only get a handful of steps
- Consider extending `DURATION_S` to 10–15 min for enough data

## Launch
```bash
torchrun --nproc_per_node=8 workloads/train_e4.py

# Override output path:
E4_OUTPUT_CSV=data/e4_custom.csv torchrun --nproc_per_node=8 workloads/train_e4.py
```

## Output
```
data/e4_telemetry.csv
```

## Dependencies
- Same as T1 (no additional packages)

## Status
**Implemented.** Script at `workloads/train_e4.py`. Not yet run — no telemetry data collected.

## Complexity
**Lowest of all evasion conditions.** Thin wrapper script + env vars. High info value per effort.

## Priority
**Second highest evasion priority** (after E1). Directly tests the most important detection signal.

# E4 — PCIe-Only Allreduce

## Goal
Force NCCL to use PCIe instead of NVLink for allreduce. The NVLink heartbeat — the strongest training signal — goes completely dark. Tests whether other signals (PCIe pattern, SM util, memory, power) can compensate.

## Evasion angle
NVLink traffic is the "smoking gun" for training. If an adversary disables NVLink for NCCL (at the cost of much slower training), NVLink-based detectors see nothing. Allreduce traffic shifts to PCIe, which is harder to distinguish from data loading or inference I/O.

## Implementation approach
No code changes needed — just environment variables. Run T1 with NCCL forced to PCIe.

### Launch command
```bash
NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 torchrun --nproc_per_node=8 workloads/train_t1.py
```

- `NCCL_P2P_DISABLE=1`: Disables NVLink/NVSwitch peer-to-peer transfers
- `NCCL_SHM_DISABLE=1`: Disables shared memory transport (forces socket/PCIe path)

### Optional: verify NVLink is actually dark
```bash
# During run, check NVLink counters
nvidia-smi dmon -s n -d 1
# Should show near-zero NVLink TX/RX
```

### Key parameters
- Model: same 3.2B GPT as T1 (no changes)
- The ONLY difference from T1 is the environment variables
- Training will be significantly slower (~5–10× due to PCIe bandwidth bottleneck)
- May need to extend `DURATION_S` to capture enough steady-state steps

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
NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 torchrun --nproc_per_node=8 workloads/train_t1.py
```

Optionally override output path:
```bash
# Modify OUTPUT_CSV in train_t1.py or add an env var override
```

## Output
```
data/e4_telemetry.csv
```

## Dependencies
- Same as T1 (no additional packages)

## Complexity
**Lowest of all evasion conditions.** Zero code changes — just env vars. High info value per effort.

## Priority
**Second highest evasion priority** (after E1). Directly tests the most important detection signal.

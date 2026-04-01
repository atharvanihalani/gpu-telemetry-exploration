# E1 — Power-Capped Training

## Goal
Run full DDP pre-training with GPU power limits throttled to inference-like levels. Tests whether power magnitude alone is a sufficient training signal, or whether NVLink + SM patterns still discriminate even at low power.

## Evasion angle
Power is often cited as the hardest signal to fake — it's measurable from facility power meters. If an adversary can train at inference-like power levels (at the cost of slower training), a power-only detector is defeated. This experiment tests what survives.

## Implementation approach
Fork `train_t1.py` → `train_e1.py`. Add power capping via pynvml before starting the training loop, restore default limits on exit.

### Key code
```python
import pynvml

pynvml.nvmlInit()
n_gpus = pynvml.nvmlDeviceGetCount()
handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n_gpus)]

# Get default TDP
default_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handles[0])  # milliwatts
target_pct = 0.22  # ~22% of TDP → inference-like power

# Set power cap
target_mw = int(default_limit * target_pct)
for h in handles:
    pynvml.nvmlDeviceSetPowerManagementLimit(h, target_mw)

# ... run training (same as T1) ...

# Restore default on exit
for h in handles:
    pynvml.nvmlDeviceSetPowerManagementLimit(h, default_limit)
```

### Power cap targets (TDP-relative)
| GPU | TDP | 22% cap | Inference range |
|---|---|---|---|
| A100 SXM4 | 400W | ~88W | 60–150W |
| H100 SXM5 | 700W | ~154W | 70–200W |

The target is to land squarely in the inference power range for the GPU. Start at ~22% of TDP, then optionally sweep: 15%, 22%, 30%, 40%.

### Key parameters
- Model: same 3.2B GPT as T1
- `POWER_CAP_PCT`: 0.22 (primary variable)
- Everything else: same as T1

## Expected telemetry signature
- **Power**: Flat at the cap (~88W A100, ~154W H100) — indistinguishable from inference by magnitude alone
- **SM util**: Lower than T1 — GPU is power-throttled, so SM clocks drop. But still non-zero and sustained.
- **Memory**: Same as T1 (~66 GB) — power cap doesn't affect memory
- **NVLink**: Same allreduce heartbeat — power cap doesn't suppress communication
- **Clock speeds**: Will drop significantly (GPU auto-throttles to stay within power budget)
- **Temperature**: Lower (less power dissipated)
- **Key question**: Memory + NVLink heartbeat survive power capping. Does a multi-signal detector still catch it?

## Hardware notes
- `nvmlDeviceSetPowerManagementLimit` requires root — should be fine on RunPod
- Power cap minimum varies by GPU. Check: `nvmlDeviceGetPowerManagementLimitConstraints(handle)` returns (min_mW, max_mW)
- Training throughput will drop significantly (~3–5× slower)
- **Important**: Always restore default limits on exit to avoid leaving the node throttled

## Launch
```bash
torchrun --nproc_per_node=8 workloads/train_e1.py
```

## Output
```
data/e1_telemetry.csv
```

## Dependencies
- Same as T1 + pynvml for power management API

## Priority
**Highest evasion priority.** This directly tests the most-cited detection signal (power).

---

## Implementation notes (2026-03-31)

### File
`workloads/train_e1.py`

### Design decisions
- **Imports GPT model class from train_t1.py** rather than duplicating ~100 lines of model code. Only the power cap logic and `main()` are new. Training loop is a local copy (not imported) because it needs to be wrapped in try/finally for power restoration.
- **Only rank 0 modifies power limits.** pynvml sees all GPUs from any process, so having all 8 ranks race to set limits would be redundant and potentially error-prone. A `dist.barrier()` after `setup_power_cap()` ensures all ranks wait for the cap to take effect before training starts.
- **Module-level `_power_state` dict** holds handles and default values so that atexit/signal handlers can access them without closures or global variables.
- **Three-layer restoration safety:**
  1. `try/finally` around `_run_training()` in `main()`
  2. `atexit.register()` for unexpected exits
  3. Signal handlers for SIGTERM and SIGINT that restore, then re-raise with default handler
- **`restore_power()` is idempotent** — tracks `_power_state["restored"]` to avoid double-restoring.
- **Respects hardware min/max constraints** via `nvmlDeviceGetPowerManagementLimitConstraints()`. The target is clamped to `[min_mw, max_mw]`.
- **Prints manual restore command at startup** so the user can recover after a `kill -9`.

### Config constant
```python
POWER_CAP_PCT = 0.22  # ~22% of TDP → ~88W on A100, ~154W on H100
```
Easy to change for sweep experiments (e.g., 0.15, 0.22, 0.30, 0.40).

### Verified
- `python -c "import workloads.train_e1"` — imports cleanly, no side effects
- AST parses without errors
- Not yet run on GPU (requires `torchrun --nproc_per_node=8 workloads/train_e1.py`)

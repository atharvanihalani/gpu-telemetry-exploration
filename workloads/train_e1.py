"""
E1 — Power-Capped Pre-Training (DDP, synthetic data)

Evasion condition: runs the same 3.2B GPT DDP training as T1 but with
GPU power limits clamped to inference-like levels (~22% of TDP).

Tests whether power magnitude alone is a sufficient training detector,
or whether NVLink heartbeat + SM patterns still discriminate under
power throttling.

Launch:
    torchrun --nproc_per_node=8 workloads/train_e1.py

Config (edit at top of file):
    POWER_CAP_PCT   fraction of default power limit to enforce (0.22 = ~88W on A100)
    DURATION_S      total wall-clock training time (default 5 min)
    WARMUP_S        initial phase excluded from "steady" analysis
    D_MODEL / N_LAYERS / N_HEADS  model architecture (same as T1)
    SEQ_LEN / BATCH_SIZE          sequence and batch dimensions
"""

import os
import sys
import time
import atexit
import signal

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn

import pynvml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workloads.collect_telemetry import TelemetryCollector

# Import model class from T1 — avoids duplicating ~100 lines of model code
from workloads.train_t1 import GPT

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
POWER_CAP_PCT = 0.22   # fraction of default power limit (~22% of TDP)

DURATION_S  = 5 * 60   # total run time
WARMUP_S    = 30        # first N seconds labelled "warmup"

# Model config — identical to T1
D_MODEL     = 3072
N_LAYERS    = 28
N_HEADS     = 24
FFN_MULT    = 4
SEQ_LEN     = 2048
BATCH_SIZE  = 4
VOCAB_SIZE  = 32000
LR          = 3e-4

OUTPUT_CSV  = "data/e1_telemetry.csv"


# ---------------------------------------------------------------------------
# Power cap management
# ---------------------------------------------------------------------------

# Module-level state so atexit/signal handlers can access it
_power_state = {
    "handles": [],
    "defaults_mw": [],
    "capped": False,
    "restored": False,
}


def setup_power_cap(local_rank, is_rank0):
    """
    Read default power limits, apply the cap, and register restoration hooks.

    Only rank 0 modifies power limits (pynvml sees all GPUs from any process).
    All ranks call this so the barrier/print logic works, but only rank 0
    actually touches the hardware.
    """
    if not is_rank0:
        return

    pynvml.nvmlInit()
    n_gpus = pynvml.nvmlDeviceGetCount()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n_gpus)]

    # Read defaults
    defaults_mw = []
    for h in handles:
        defaults_mw.append(pynvml.nvmlDeviceGetPowerManagementLimit(h))

    # Store in module-level state for restoration hooks
    _power_state["handles"] = handles
    _power_state["defaults_mw"] = defaults_mw

    # Print manual restore command before changing anything
    default_watts = defaults_mw[0] // 1000
    print(f"[E1] Default power limit: {default_watts}W per GPU")
    print(f"[E1] If power limits get stuck, run: nvidia-smi -pl {default_watts}")

    # Apply cap, respecting hardware min/max constraints
    for i, h in enumerate(handles):
        min_mw, max_mw = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)
        target_mw = int(defaults_mw[i] * POWER_CAP_PCT)
        target_mw = max(target_mw, min_mw)   # respect hardware minimum
        target_mw = min(target_mw, max_mw)   # respect hardware maximum
        pynvml.nvmlDeviceSetPowerManagementLimit(h, target_mw)
        actual_w = target_mw / 1000
        print(f"[E1] GPU {i}: power limit {defaults_mw[i]/1000:.0f}W → {actual_w:.0f}W "
              f"({POWER_CAP_PCT*100:.0f}% of default, hw min={min_mw/1000:.0f}W)")

    _power_state["capped"] = True
    print(f"[E1] Power cap applied: {POWER_CAP_PCT*100:.0f}% of TDP across {n_gpus} GPUs")

    # Register restoration hooks
    atexit.register(_restore_power_atexit)
    signal.signal(signal.SIGTERM, _restore_power_signal)
    signal.signal(signal.SIGINT, _restore_power_signal)


def restore_power():
    """Restore default power limits. Safe to call multiple times."""
    if _power_state["restored"] or not _power_state["capped"]:
        return

    handles = _power_state["handles"]
    defaults_mw = _power_state["defaults_mw"]

    restored_count = 0
    for h, d in zip(handles, defaults_mw):
        try:
            pynvml.nvmlDeviceSetPowerManagementLimit(h, d)
            restored_count += 1
        except Exception as e:
            # Best effort — print so the user knows to fix manually
            print(f"[E1] WARNING: failed to restore power limit: {e}")

    _power_state["restored"] = True
    print(f"[E1] Power limits restored to defaults ({restored_count}/{len(handles)} GPUs)")

    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass


def _restore_power_atexit():
    """atexit hook — backup for unexpected exits."""
    restore_power()


def _restore_power_signal(signum, frame):
    """Signal handler for SIGTERM/SIGINT — restore power then re-raise."""
    restore_power()
    # Re-raise the signal with default handler so the process exits properly
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # DDP init
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    is_rank0 = rank == 0

    if is_rank0:
        print(f"[E1] Power-capped training — DDP world_size={world_size}")
        print(f"[E1] POWER_CAP_PCT={POWER_CAP_PCT} ({POWER_CAP_PCT*100:.0f}% of TDP)")
        print(f"[E1] Model: d_model={D_MODEL}, n_layers={N_LAYERS}, n_heads={N_HEADS}")
        print(f"[E1] Sequence length={SEQ_LEN}, batch/GPU={BATCH_SIZE}")

    # Apply power cap BEFORE any training (rank 0 only)
    setup_power_cap(local_rank, is_rank0)

    # Synchronize so all ranks wait for power cap to be applied
    dist.barrier()

    try:
        _run_training(rank, world_size, local_rank, device, is_rank0)
    finally:
        # Restore power limits no matter what happens
        if is_rank0:
            restore_power()

    dist.destroy_process_group()


def _run_training(rank, world_size, local_rank, device, is_rank0):
    """Training loop — identical to T1 except output CSV path."""

    # Telemetry — only rank 0 collects (pynvml sees all GPUs)
    collector = None
    if is_rank0:
        collector = TelemetryCollector(OUTPUT_CSV)
        collector.start()
        collector.set_phase("warmup")

    # Build model (same architecture as T1)
    model = GPT(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        ffn_mult=FFN_MULT,
        seq_len=SEQ_LEN,
    ).to(device)

    # Count params (rank 0 only)
    if is_rank0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[E1] Parameters: {n_params / 1e9:.2f}B")

    # bf16 AMP
    model = model.to(torch.bfloat16)
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn   = nn.CrossEntropyLoss()

    # Synthetic data — random token IDs
    def get_batch():
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        return x, y

    # Training loop
    step      = 0
    t_start   = time.time()
    switched  = False

    if is_rank0:
        print(f"[E1] Training for {DURATION_S}s (warmup={WARMUP_S}s) ...")

    while True:
        elapsed = time.time() - t_start
        if elapsed >= DURATION_S:
            break

        # Switch phase label after warmup
        if not switched and elapsed >= WARMUP_S:
            if is_rank0 and collector:
                collector.set_phase("steady")
            switched = True

        x, y = get_batch()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = loss_fn(
                logits.reshape(-1, VOCAB_SIZE),
                y.reshape(-1),
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()          # allreduce happens here (DDP hook)
        optimizer.step()

        step += 1
        if is_rank0 and step % 10 == 0:
            print(f"  step={step:4d}  loss={loss.item():.4f}  "
                  f"elapsed={elapsed:.0f}s")

    if is_rank0:
        if collector:
            collector.set_phase("cooldown")
            time.sleep(5)
            collector.stop()
        print(f"[E1] Done. {step} steps in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()

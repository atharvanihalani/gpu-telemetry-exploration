"""
E4 — PCIe-Only Allreduce (DDP pre-training with NVLink disabled)

Identical to T1 (large pre-training) except NCCL is forced to use PCIe
instead of NVLink for allreduce communication. This kills the NVLink
heartbeat — the single strongest training signal — and shifts gradient
sync traffic to PCIe.

Evasion angle: if an adversary disables NVLink for NCCL (at the cost of
much slower training), NVLink-based detectors see nothing. The question
is whether other signals (PCIe pattern, SM util, memory, power) can
compensate.

Environment variables set before NCCL init:
    NCCL_P2P_DISABLE=1   — disables NVLink/NVSwitch peer-to-peer
    NCCL_SHM_DISABLE=1   — disables shared memory, forces socket/PCIe

Launch:
    torchrun --nproc_per_node=8 workloads/train_e4.py

Verify NVLink is dark during the run:
    nvidia-smi dmon -s n -d 1   # should show near-zero NVLink TX/RX

Config overrides vs T1:
    DURATION_S   10 min (vs 5 min) — steps are much slower over PCIe
    OUTPUT_CSV   data/e4_telemetry.csv (env var E4_OUTPUT_CSV to override)
"""

import os

# -----------------------------------------------------------------------
# Force NCCL to PCIe — MUST be set before any NCCL initialization
# (i.e. before dist.init_process_group)
# -----------------------------------------------------------------------
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"

import sys
import time
import math

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workloads.collect_telemetry import TelemetryCollector

# Import model architecture from T1 — same model, different comms backend
from workloads.train_t1 import (
    GPT,
    D_MODEL,
    N_LAYERS,
    N_HEADS,
    FFN_MULT,
    SEQ_LEN,
    BATCH_SIZE,
    VOCAB_SIZE,
    LR,
    WARMUP_S,
)

# ---------------------------------------------------------------------------
# E4-specific config overrides
# ---------------------------------------------------------------------------
DURATION_S = 10 * 60   # 10 min (doubled from T1's 5 min — PCIe allreduce is slow)
OUTPUT_CSV = os.environ.get("E4_OUTPUT_CSV", "data/e4_telemetry.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # DDP init — NCCL will use PCIe path due to env vars set above
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    is_rank0 = rank == 0

    if is_rank0:
        print("=" * 60)
        print("E4 — PCIe-Only Allreduce (NVLink disabled)")
        print("=" * 60)
        print(f"  NCCL_P2P_DISABLE = {os.environ.get('NCCL_P2P_DISABLE')}")
        print(f"  NCCL_SHM_DISABLE = {os.environ.get('NCCL_SHM_DISABLE')}")
        print(f"  DDP world_size={world_size}, d_model={D_MODEL}, "
              f"n_layers={N_LAYERS}, n_heads={N_HEADS}")
        print(f"  Sequence length={SEQ_LEN}, batch/GPU={BATCH_SIZE}")
        print(f"  Duration={DURATION_S}s, output={OUTPUT_CSV}")
        print()

        # Warn about expected slowdown
        print("NOTE: Training will be significantly slower (~5-10x) due to")
        print("PCIe bandwidth bottleneck for allreduce. This is expected.")
        print()

    # Telemetry — only rank 0 collects (DCGM sees all GPUs)
    collector = None
    if is_rank0:
        collector = TelemetryCollector(OUTPUT_CSV)
        collector.start()
        collector.set_phase("warmup")

    # Build model (identical to T1)
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
        print(f"Parameters: {n_params / 1e9:.2f}B")

    # bf16 AMP
    model = model.to(torch.bfloat16)
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn   = nn.CrossEntropyLoss()

    # Synthetic data — random token IDs (identical to T1)
    def get_batch():
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        return x, y

    # Training loop (identical to T1)
    step      = 0
    t_start   = time.time()
    switched  = False

    if is_rank0:
        print(f"Training for {DURATION_S}s (warmup={WARMUP_S}s) ...")

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
        loss.backward()          # allreduce happens here (DDP hook) — over PCIe!
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
        print(f"Done. {step} steps in {time.time() - t_start:.1f}s")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

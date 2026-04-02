"""
E5 — Smoothed Allreduce (Large NCCL Buffer, Ring Algorithm)

Identical to T1 (large pre-training) except NCCL is tuned to spread the
allreduce communication over a wider window at lower peak bandwidth. The
total bytes transferred are the same, but the temporal profile changes:
sharp NVLink bursts become low, continuous traffic.

Evasion angle: a detector looking for periodic NVLink spikes (the allreduce
"heartbeat") might miss a smoothed-out version where traffic is spread
across a longer window. If the allreduce window stretches beyond the
detector's polling interval (e.g. >1s at 1Hz sampling), the heartbeat
becomes invisible — it looks like continuous background traffic.

Environment variables set before NCCL init:
    NCCL_ALGO=Ring           — force ring algorithm (predictable, stretchable)
    NCCL_BUFFSIZE=16777216   — 16MB buffer (default 4MB) = longer transfer window
    NCCL_MAX_NCHANNELS=4     — 4 channels (default auto ~8-12) = lower peak rate

Launch:
    torchrun --nproc_per_node=8 workloads/train_e5.py

Verify smoothing during the run:
    nvidia-smi dmon -s n -d 1   # NVLink should show lower peak, longer duration
    dcgmi dmon -e 409,410,411,412,413,414,415,416,417,418,419,420 -d 500
                                # Sub-second DCGM to see the temporal spread

Config overrides vs T1:
    DURATION_S   10 min (vs 5 min) — smoothed allreduce is slower per step
    OUTPUT_CSV   data/e5_telemetry.csv (env var E5_OUTPUT_CSV to override)
"""

import os

# -----------------------------------------------------------------------
# NCCL smoothing config — MUST be set before any NCCL initialization
# (i.e. before dist.init_process_group / any torch.distributed import)
# -----------------------------------------------------------------------

# Tunable constants — edit these to explore the smoothing space:
#   Mild:       BUFFSIZE=16MB,  MAX_NCHANNELS=4   (slightly broader peaks)
#   Medium:     BUFFSIZE=128MB, MAX_NCHANNELS=2   (notably smoothed)
#   Aggressive: BUFFSIZE=128MB, MAX_NCHANNELS=1   (maximum smoothing)
NCCL_ALGO = "Ring"
NCCL_BUFFSIZE = "33554432"    # 32MB (default is 4MB)
NCCL_MAX_NCHANNELS = "2"      # 2 channels (default: auto ~8-12)

os.environ["NCCL_ALGO"] = NCCL_ALGO
os.environ["NCCL_BUFFSIZE"] = NCCL_BUFFSIZE
os.environ["NCCL_MAX_NCHANNELS"] = NCCL_MAX_NCHANNELS

import sys
import time
import math

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workloads.collect_telemetry import TelemetryCollector

# Import model architecture from T1 — same model, different NCCL tuning
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
# E5-specific config overrides
# ---------------------------------------------------------------------------
DURATION_S = 10 * 60   # 10 min (doubled from T1's 5 min — smoothed allreduce is slower)
OUTPUT_CSV = os.environ.get("E5_OUTPUT_CSV", "data/e5_telemetry.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # DDP init — NCCL will use ring algo with large buffers due to env vars above
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    is_rank0 = rank == 0

    if is_rank0:
        print("=" * 60)
        print("E5 — Smoothed Allreduce (Large NCCL Buffer, Ring Algorithm)")
        print("=" * 60)
        print(f"  NCCL_ALGO           = {os.environ.get('NCCL_ALGO')}")
        print(f"  NCCL_BUFFSIZE       = {os.environ.get('NCCL_BUFFSIZE')}"
              f" ({int(os.environ.get('NCCL_BUFFSIZE', 0)) / 1024 / 1024:.0f}MB)")
        print(f"  NCCL_MAX_NCHANNELS  = {os.environ.get('NCCL_MAX_NCHANNELS')}")
        print(f"  DDP world_size={world_size}, d_model={D_MODEL}, "
              f"n_layers={N_LAYERS}, n_heads={N_HEADS}")
        print(f"  Sequence length={SEQ_LEN}, batch/GPU={BATCH_SIZE}")
        print(f"  Duration={DURATION_S}s, output={OUTPUT_CSV}")
        print()

        # Explain expected behavior
        print("NOTE: Training will be slower due to single-channel ring allreduce.")
        print("NVLink traffic should be spread over a wider window at lower peak BW.")
        print("Compare with T1 NVLink heatmap to see smoothing effect.")
        print()

    # Telemetry — only rank 0 collects (pynvml sees all GPUs)
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
        loss.backward()          # allreduce happens here (DDP hook) — smoothed!
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

"""
T10 — Multi-Node DDP Pre-Training (16-GPU, 2 nodes)

First multi-node condition. Same 3.2B GPT as T1, but DDP across 16 GPUs
on 2 nodes connected via InfiniBand. Allreduce now has two tiers:
  - Intra-node: NVLink (900 GB/s per H100, visible in DCGM)
  - Inter-node: InfiniBand (8x NDR 400G, NOT visible in DCGM)

Each node's LOCAL_RANK=0 process runs an independent DCGM telemetry
collector, producing one CSV per node. DCGM is node-local — it only
sees the 8 GPUs on its own machine.

Expected telemetry vs T1 (single-node):
  - Power: similar sustained ~650-700W per GPU
  - NVLink: heartbeat still visible (intra-node allreduce portion)
  - Step time: slightly slower (IB latency for inter-node gradient sync)
  - IB traffic: invisible in DCGM; would need sysfs counters

Launch (run on BOTH nodes, within ~60s of each other):

    # Node 0:
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \\
      --master_addr=<NODE0_PRIVATE_IP> --master_port=29500 \\
      workloads/train_t10.py

    # Node 1:
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \\
      --master_addr=<NODE0_PRIVATE_IP> --master_port=29500 \\
      workloads/train_t10.py

Output:
    data/t10_node0_telemetry.csv
    data/t10_node1_telemetry.csv
"""

import os
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workloads.collect_telemetry import TelemetryCollector
from workloads.collect_ib import IBCollector
from workloads.collect_bmc import BMCCollector

# Import model architecture and config from T1
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
# T10-specific config
# ---------------------------------------------------------------------------
DURATION_S = 5 * 60
OUTPUT_CSV_TEMPLATE = "data/t10_node{}_telemetry.csv"
OUTPUT_IB_TEMPLATE  = "data/t10_node{}_ib.csv"
OUTPUT_BMC_TEMPLATE = "data/t10_node{}_bmc.csv"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    nproc_per_node = torch.cuda.device_count()
    node_rank = rank // nproc_per_node
    is_rank0 = rank == 0
    is_local_rank0 = local_rank == 0

    if is_rank0:
        print(f"T10 — Multi-Node DDP")
        print(f"  world_size={world_size}, nodes={world_size // nproc_per_node}, "
              f"gpus_per_node={nproc_per_node}")
        print(f"  d_model={D_MODEL}, n_layers={N_LAYERS}, n_heads={N_HEADS}")
        print(f"  seq_len={SEQ_LEN}, batch/GPU={BATCH_SIZE}")

    # Telemetry — LOCAL_RANK=0 on EACH node starts collectors
    collector = None
    ib_collector = None
    bmc_collector = None
    if is_local_rank0 and not os.environ.get("TELEMETRY_DISABLED"):
        collector = TelemetryCollector(OUTPUT_CSV_TEMPLATE.format(node_rank))
        collector.start()
        collector.set_phase("warmup")

        ib_collector = IBCollector(OUTPUT_IB_TEMPLATE.format(node_rank))
        ib_collector.start()
        ib_collector.set_phase("warmup")

        bmc_collector = BMCCollector(OUTPUT_BMC_TEMPLATE.format(node_rank))
        bmc_collector.start()
        bmc_collector.set_phase("warmup")

    # Build model (identical to T1)
    model = GPT(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        ffn_mult=FFN_MULT,
        seq_len=SEQ_LEN,
    ).to(device)

    if is_rank0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params / 1e9:.2f}B")

    model = model.to(torch.bfloat16)
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn   = nn.CrossEntropyLoss()

    def get_batch():
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        return x, y

    # Training loop
    step     = 0
    t_start  = time.time()
    switched = False

    if is_rank0:
        print(f"  Training for {DURATION_S}s (warmup={WARMUP_S}s) ...")

    while True:
        elapsed = time.time() - t_start
        if elapsed >= DURATION_S:
            break

        if not switched and elapsed >= WARMUP_S:
            if collector:
                collector.set_phase("steady")
            if ib_collector:
                ib_collector.set_phase("steady")
            if bmc_collector:
                bmc_collector.set_phase("steady")
            switched = True

        x, y = get_batch()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = loss_fn(
                logits.reshape(-1, VOCAB_SIZE),
                y.reshape(-1),
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        step += 1
        if is_rank0 and step % 10 == 0:
            print(f"  step={step:4d}  loss={loss.item():.4f}  "
                  f"elapsed={elapsed:.0f}s")

    if collector:
        collector.set_phase("cooldown")
    if ib_collector:
        ib_collector.set_phase("cooldown")
    if bmc_collector:
        bmc_collector.set_phase("cooldown")
    if collector:
        time.sleep(5)
        collector.stop()
    if ib_collector:
        ib_collector.stop()
    if bmc_collector:
        bmc_collector.stop()

    if is_rank0:
        print(f"Done. {step} steps in {time.time() - t_start:.1f}s")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""
T15 — Full FSDP (ZeRO-3) Multi-Node Training (16-GPU, 2 nodes)

Full Sharded Data Parallelism across all 16 GPUs on 2 nodes. Same 3.2B GPT
as T1, wrapped in FSDP with per-TransformerBlock sharding.

FSDP communication pattern (distinct from DDP):
  - Forward: per-layer all-gather to reconstruct full parameters (NVLink + IB)
  - Backward: per-layer reduce-scatter to distribute gradients (NVLink + IB)
  - Both NVLink and IB are active CONTINUOUSLY throughout the step,
    not in periodic bursts like DDP's end-of-step allreduce

This is the first condition with continuous, non-periodic IB traffic. DDP
(T10) has periodic IB bursts at step boundaries; TP+DP (T11) has continuous
NVLink but periodic IB. FSDP spreads communication across the entire step
on both interconnects.

No CPU offload (unlike T6) — we want pure FSDP communication patterns.
No torch.amp.autocast — FSDP's MixedPrecision handles dtype casting.

Launch (run on BOTH nodes within ~60s):

    # Node 0:
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \\
      --master_addr=<NODE0_PRIVATE_IP> --master_port=29500 \\
      workloads/train_t15.py

    # Node 1:
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \\
      --master_addr=<NODE0_PRIVATE_IP> --master_port=29500 \\
      workloads/train_t15.py

Output (per node):
    data/t15_node{N}_telemetry.csv  (DCGM, 10Hz)
    data/t15_node{N}_ib.csv         (InfiniBand, 10Hz)
    data/t15_node{N}_bmc.csv        (BMC, 2s)
"""

import os
import sys
import time
import functools

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workloads.collect_telemetry import TelemetryCollector
from workloads.collect_ib import IBCollector
from workloads.collect_bmc import BMCCollector

# Import model architecture and config from T1
from workloads.train_t1 import (
    GPT,
    TransformerBlock,
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
# T15-specific config
# ---------------------------------------------------------------------------
DURATION_S = 5 * 60
OUTPUT_CSV_TEMPLATE = "data/t15_node{}_telemetry.csv"
OUTPUT_IB_TEMPLATE  = "data/t15_node{}_ib.csv"
OUTPUT_BMC_TEMPLATE = "data/t15_node{}_bmc.csv"


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
        print(f"T15 — Full FSDP (ZeRO-3) Multi-Node")
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

    # Build model on CPU first — FSDP with sync_module_states=True
    # broadcasts rank-0 weights to all ranks during wrapping
    if is_rank0:
        print("Building model on CPU ...")
    model = GPT(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        ffn_mult=FFN_MULT,
        seq_len=SEQ_LEN,
    )

    if is_rank0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params / 1e9:.2f}B")

    # -------------------------------------------------------------------
    # FSDP wrapping — no CPU offload, pure sharded communication
    # -------------------------------------------------------------------

    # Auto-wrap: each TransformerBlock becomes its own FSDP unit
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    # Mixed precision: bf16 everywhere (matches T6, no autocast needed)
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=local_rank,
        sync_module_states=True,
        use_orig_params=False,
    )

    if is_rank0:
        print("  FSDP mode: FULL_SHARD (no CPU offload)")
        torch.cuda.synchronize()
        mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
        mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        print(f"  GPU memory after setup: {mem_alloc:.2f} GB allocated, "
              f"{mem_reserved:.2f} GB reserved")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn   = nn.CrossEntropyLoss()

    # Synthetic data — random token IDs
    def get_batch():
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        return x, y

    # -------------------------------------------------------------------
    # Training loop (no autocast — FSDP MixedPrecision handles dtypes)
    # -------------------------------------------------------------------
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

        # Forward — FSDP all-gathers params per layer
        logits = model(x)
        loss = loss_fn(
            logits.reshape(-1, VOCAB_SIZE),
            y.reshape(-1),
        )

        # Backward — FSDP reduce-scatters gradients per layer
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        step += 1
        if is_rank0 and step % 10 == 0:
            mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
            print(f"  step={step:4d}  loss={loss.item():.4f}  "
                  f"elapsed={elapsed:.0f}s  gpu_mem={mem_alloc:.2f}GB")

    # Cooldown
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

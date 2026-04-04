"""
T11 — TP+DP Hybrid Training (8-way TP within node, 2-way DP across nodes)

Realistic frontier training configuration: tensor parallelism (TP) shards
each layer across all 8 GPUs within a node, while data parallelism (DP)
replicates the model across 2 nodes. This produces:

  - Intra-node NVLink: CONTINUOUS high bandwidth (TP all-reduces every layer,
    forward and backward) — NOT periodic heartbeat
  - Inter-node InfiniBand: PERIODIC DP gradient allreduce every step

This directly challenges the nvlink_autocorr classifier rule, which detects
periodic NVLink bursts. TP's continuous traffic has no periodicity.

Uses PyTorch native tensor parallelism (parallelize_module + DeviceMesh).

Launch (run on BOTH nodes within ~60s):

    # Node 0:
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \\
      --master_addr=<NODE0_PRIVATE_IP> --master_port=29500 \\
      workloads/train_t11.py

    # Node 1:
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \\
      --master_addr=<NODE0_PRIVATE_IP> --master_port=29500 \\
      workloads/train_t11.py

Output (per node):
    data/t11_node{N}_telemetry.csv  (DCGM, 10Hz)
    data/t11_node{N}_ib.csv         (InfiniBand, 10Hz)
    data/t11_node{N}_bmc.csv        (BMC, 2s)
"""

import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed._composable.replicate import replicate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workloads.collect_telemetry import TelemetryCollector
from workloads.collect_ib import IBCollector
from workloads.collect_bmc import BMCCollector

# Import config from T1 (but NOT the model — we need a TP-aware version)
from workloads.train_t1 import (
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
# TP-aware model (forward uses local head count after sharding)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Attention with TP-safe forward: infers head count from sharded QKV output."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        qkv_out = self.qkv(x)
        # After ColwiseParallel, output dim is 3*d_model/tp_size.
        # Infer local head count from actual output size.
        local_qkv_dim = qkv_out.shape[-1]
        local_heads = local_qkv_dim // (3 * self.head_dim)
        qkv = qkv_out.reshape(B, T, 3, local_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )
        y = y.transpose(1, 2).reshape(B, T, local_heads * self.head_dim)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_mult):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model, bias=False),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, ffn_mult, seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_mult) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

# ---------------------------------------------------------------------------
# T11-specific config
# ---------------------------------------------------------------------------
DURATION_S = 5 * 60
OUTPUT_CSV_TEMPLATE = "data/t11_node{}_telemetry.csv"
OUTPUT_IB_TEMPLATE  = "data/t11_node{}_ib.csv"
OUTPUT_BMC_TEMPLATE = "data/t11_node{}_bmc.csv"


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
    n_nodes = world_size // nproc_per_node
    is_rank0 = rank == 0
    is_local_rank0 = local_rank == 0

    if is_rank0:
        print(f"T11 — TP+DP Hybrid Training")
        print(f"  world_size={world_size}, nodes={n_nodes}, "
              f"gpus_per_node={nproc_per_node}")
        print(f"  TP={nproc_per_node}-way (within node, NVLink)")
        print(f"  DP={n_nodes}-way (across nodes, InfiniBand)")
        print(f"  d_model={D_MODEL}, n_layers={N_LAYERS}, n_heads={N_HEADS}")
        print(f"  seq_len={SEQ_LEN}, batch/GPU={BATCH_SIZE}")

    # ------------------------------------------------------------------
    # Telemetry — LOCAL_RANK=0 on EACH node starts collectors
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 2D Device Mesh: (DP across nodes, TP within node)
    # ------------------------------------------------------------------
    mesh = init_device_mesh(
        "cuda",
        mesh_shape=(n_nodes, nproc_per_node),
        mesh_dim_names=("dp", "tp"),
    )
    tp_mesh = mesh["tp"]
    dp_mesh = mesh["dp"]

    if is_rank0:
        print(f"  DeviceMesh: dp={n_nodes}, tp={nproc_per_node}")

    # ------------------------------------------------------------------
    # Build model and apply TP + DP
    # ------------------------------------------------------------------
    model = GPT(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        ffn_mult=FFN_MULT,
        seq_len=SEQ_LEN,
    ).to(device).to(torch.bfloat16)

    if is_rank0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters (pre-shard): {n_params / 1e9:.2f}B")

    # Apply tensor parallelism to each transformer block
    for i, block in enumerate(model.blocks):
        # Attention: QKV is column-parallel, output proj is row-parallel
        parallelize_module(block.attn, tp_mesh, {
            "qkv": ColwiseParallel(),
            "proj": RowwiseParallel(),
        })
        # FFN: up-projection (index 0) is column-parallel,
        #      down-projection (index 2) is row-parallel
        #      (index 1 is GELU, not a linear layer)
        parallelize_module(block.ffn, tp_mesh, {
            "0": ColwiseParallel(),
            "2": RowwiseParallel(),
        })

    # Apply DP replication across nodes
    replicate(model, device_mesh=dp_mesh)

    if is_rank0:
        local_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters (post-shard, per GPU): {local_params / 1e6:.0f}M")
        print(f"  TP shards each layer across {nproc_per_node} GPUs")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn   = nn.CrossEntropyLoss()

    def get_batch():
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        return x, y

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
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
        loss.backward()   # TP all-reduces (NVLink) + DP allreduce (IB)
        optimizer.step()

        step += 1
        if is_rank0 and step % 10 == 0:
            print(f"  step={step:4d}  loss={loss.item():.4f}  "
                  f"elapsed={elapsed:.0f}s")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
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

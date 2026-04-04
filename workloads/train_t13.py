"""
T13 — TP+PP Hybrid Training (8-way TP within node, 2-stage PP across nodes)

Combines 8-way Tensor Parallelism within each node with 2-stage Pipeline
Parallelism across 2 nodes. No data parallelism. Each node runs one
pipeline stage:

  - Node 0 (stage 0): embedding + blocks 0-13 (first half), TP across 8 GPUs
  - Node 1 (stage 1): blocks 14-27 + ln_f + head (second half), TP across 8 GPUs

Communication pattern:
  - Intra-node NVLink: CONTINUOUS high bandwidth (TP all-reduces every layer,
    forward and backward) — same as T11
  - Inter-node InfiniBand: PERIODIC P2P transfers (pipeline activations),
    NOT allreduce. Only local_rank=0 on each node does IB send/recv, then
    broadcasts within TP group via NVLink.

Key differentiator from T10/T11: IB traffic is **point-to-point** (one sender,
one receiver) rather than **collective** (allreduce). First condition with no
allreduce anywhere.

GPipe schedule with 4 micro-batches. Bubble fraction = (2-1)/(2+4-1) = 20%.

Uses PyTorch native tensor parallelism (parallelize_module + DeviceMesh).
Manual dist.send/dist.recv for pipeline parallelism (not PiPPy).

Launch (run on BOTH nodes within ~60s):

    # Node 0:
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \\
      --master_addr=<NODE0_PRIVATE_IP> --master_port=29500 \\
      workloads/train_t13.py

    # Node 1:
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \\
      --master_addr=<NODE0_PRIVATE_IP> --master_port=29500 \\
      workloads/train_t13.py

Output (per node):
    data/t13_node{N}_telemetry.csv  (DCGM, 10Hz)
    data/t13_node{N}_ib.csv         (InfiniBand, 10Hz)
    data/t13_node{N}_bmc.csv        (BMC, 2s)
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workloads.collect_telemetry import TelemetryCollector
from workloads.collect_ib import IBCollector
from workloads.collect_bmc import BMCCollector

from workloads.train_t1 import (
    D_MODEL,
    N_LAYERS,
    N_HEADS,
    FFN_MULT,
    SEQ_LEN,
    VOCAB_SIZE,
    LR,
    WARMUP_S,
)

# ---------------------------------------------------------------------------
# T13-specific config
# ---------------------------------------------------------------------------
DURATION_S      = 5 * 60
BATCH_SIZE      = 16       # total per step, split into micro-batches
N_MICROBATCHES  = 4
MICRO_BATCH     = BATCH_SIZE // N_MICROBATCHES  # 4

LAYERS_PER_STAGE = N_LAYERS // 2  # 14

OUTPUT_CSV_TEMPLATE = "data/t13_node{}_telemetry.csv"
OUTPUT_IB_TEMPLATE  = "data/t13_node{}_ib.csv"
OUTPUT_BMC_TEMPLATE = "data/t13_node{}_bmc.csv"


# ---------------------------------------------------------------------------
# TP-aware model components (same as T11)
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


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

class GPTStage0(nn.Module):
    """Embedding + first 14 transformer blocks."""

    def __init__(self, vocab_size, d_model, n_layers_stage, n_heads, ffn_mult, seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_mult)
            for _ in range(n_layers_stage)
        ])

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return x  # (B, T, D_MODEL)


class GPTStage1(nn.Module):
    """Last 14 transformer blocks + ln_f + output head."""

    def __init__(self, vocab_size, d_model, n_layers_stage, n_heads, ffn_mult):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_mult)
            for _ in range(n_layers_stage)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, hidden):
        x = hidden
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)  # (B, T, VOCAB_SIZE)


# ---------------------------------------------------------------------------
# Helper: apply TP sharding to a stage's blocks
# ---------------------------------------------------------------------------

def apply_tp_sharding(model, tp_mesh):
    """Apply ColwiseParallel/RowwiseParallel to attention and FFN in each block."""
    for block in model.blocks:
        parallelize_module(block.attn, tp_mesh, {
            "qkv": ColwiseParallel(),
            "proj": RowwiseParallel(),
        })
        parallelize_module(block.ffn, tp_mesh, {
            "0": ColwiseParallel(),
            "2": RowwiseParallel(),
        })


# ---------------------------------------------------------------------------
# Helper: broadcast within TP group
# ---------------------------------------------------------------------------

def broadcast_within_tp(tensor, tp_group, src_global_rank):
    """Broadcast tensor from local_rank=0 to all ranks in the TP group."""
    dist.broadcast(tensor, src=src_global_rank, group=tp_group)


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

    # Pipeline stage assignment: node 0 = stage 0, node 1 = stage 1
    pp_stage = node_rank

    # Global rank of local_rank=0 on each node (for IB send/recv)
    stage0_lr0_rank = 0                     # node 0, local_rank 0
    stage1_lr0_rank = nproc_per_node        # node 1, local_rank 0

    # Global rank of local_rank=0 on this node (for TP broadcast src)
    my_lr0_global_rank = node_rank * nproc_per_node

    if is_rank0:
        print(f"T13 — TP+PP Hybrid Training")
        print(f"  world_size={world_size}, nodes={n_nodes}, "
              f"gpus_per_node={nproc_per_node}")
        print(f"  TP={nproc_per_node}-way (within node, NVLink)")
        print(f"  PP=2-stage (across nodes, InfiniBand)")
        print(f"  GPipe: {N_MICROBATCHES} micro-batches, "
              f"micro_batch_size={MICRO_BATCH}")
        print(f"  Stage 0: embedding + layers 0-{LAYERS_PER_STAGE-1}")
        print(f"  Stage 1: layers {LAYERS_PER_STAGE}-{N_LAYERS-1} + ln_f + head")
        print(f"  d_model={D_MODEL}, n_layers={N_LAYERS}, n_heads={N_HEADS}")
        print(f"  seq_len={SEQ_LEN}, batch_size={BATCH_SIZE}")

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
    # 2D Device Mesh: (PP across nodes, TP within node)
    # ------------------------------------------------------------------
    mesh = init_device_mesh(
        "cuda",
        mesh_shape=(n_nodes, nproc_per_node),
        mesh_dim_names=("pp", "tp"),
    )
    tp_mesh = mesh["tp"]

    # TP process group for this node (for broadcast_within_tp)
    tp_group = tp_mesh.get_group()

    if is_rank0:
        print(f"  DeviceMesh: pp={n_nodes}, tp={nproc_per_node}")

    # ------------------------------------------------------------------
    # Build model stage and apply TP
    # ------------------------------------------------------------------
    if pp_stage == 0:
        model = GPTStage0(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            n_layers_stage=LAYERS_PER_STAGE,
            n_heads=N_HEADS,
            ffn_mult=FFN_MULT,
            seq_len=SEQ_LEN,
        ).to(device).to(torch.bfloat16)
    else:
        model = GPTStage1(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            n_layers_stage=LAYERS_PER_STAGE,
            n_heads=N_HEADS,
            ffn_mult=FFN_MULT,
        ).to(device).to(torch.bfloat16)

    if is_rank0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters (pre-shard, stage {pp_stage}): {n_params / 1e9:.2f}B")

    apply_tp_sharding(model, tp_mesh)

    if is_rank0:
        local_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters (post-shard, per GPU): {local_params / 1e6:.0f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn   = nn.CrossEntropyLoss()

    # Activation shape for inter-stage transfers: (MICRO_BATCH, SEQ_LEN, D_MODEL)
    act_shape = (MICRO_BATCH, SEQ_LEN, D_MODEL)

    def get_batch():
        """Generate a full batch of synthetic data, return list of micro-batches."""
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        x_mbs = list(x.split(MICRO_BATCH, dim=0))
        y_mbs = list(y.split(MICRO_BATCH, dim=0))
        return x_mbs, y_mbs

    # ------------------------------------------------------------------
    # GPipe step function
    # ------------------------------------------------------------------

    def gpipe_step():
        """Execute one GPipe step with N_MICROBATCHES micro-batches.

        Returns average loss (stage 1 only, 0.0 on stage 0).
        """
        x_mbs, y_mbs = get_batch()

        if pp_stage == 0:
            # Stage 0 stores hidden activations for backward
            saved_inputs = [None] * N_MICROBATCHES   # input token ids (no grad needed)
            saved_hiddens = [None] * N_MICROBATCHES  # output hidden states (need grad)

            # === FORWARD: micro-batches 0..3 ===
            for mb in range(N_MICROBATCHES):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    hidden = model(x_mbs[mb])  # (MICRO_BATCH, SEQ_LEN, D_MODEL)

                # Save for backward
                saved_inputs[mb] = x_mbs[mb]
                saved_hiddens[mb] = hidden  # retains grad

                # Send hidden to stage 1 via IB (only local_rank=0)
                if is_local_rank0:
                    # Explicitly cast to bf16 for send (autocast doesn't cover comms)
                    send_buf = hidden.detach().to(torch.bfloat16).contiguous()
                    dist.send(send_buf, dst=stage1_lr0_rank, tag=mb)

            # === BACKWARD: micro-batches 3..0 (reversed) ===
            for mb in reversed(range(N_MICROBATCHES)):
                # Receive grad from stage 1 via IB (only local_rank=0)
                grad_buf = torch.empty(act_shape, dtype=torch.bfloat16, device=device)
                if is_local_rank0:
                    dist.recv(grad_buf, src=stage1_lr0_rank,
                              tag=N_MICROBATCHES + mb)

                # Broadcast received grad to all TP ranks
                broadcast_within_tp(grad_buf, tp_group, my_lr0_global_rank)

                # Backward through stage 0
                saved_hiddens[mb].backward(grad_buf)

            return 0.0  # no loss on stage 0

        else:
            # Stage 1 stores activations and losses for backward
            saved_hiddens = [None] * N_MICROBATCHES
            saved_losses = [None] * N_MICROBATCHES

            # === FORWARD: micro-batches 0..3 ===
            for mb in range(N_MICROBATCHES):
                # Receive hidden from stage 0 via IB (only local_rank=0)
                recv_buf = torch.empty(act_shape, dtype=torch.bfloat16, device=device)
                if is_local_rank0:
                    dist.recv(recv_buf, src=stage0_lr0_rank, tag=mb)

                # Broadcast received hidden to all TP ranks
                broadcast_within_tp(recv_buf, tp_group, my_lr0_global_rank)

                # Enable gradient flow through received activation
                recv_buf = recv_buf.requires_grad_(True)
                saved_hiddens[mb] = recv_buf

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(recv_buf)
                    loss = loss_fn(
                        logits.reshape(-1, VOCAB_SIZE),
                        y_mbs[mb].reshape(-1),
                    )

                # Scale loss by N_MICROBATCHES for gradient accumulation
                saved_losses[mb] = loss / N_MICROBATCHES

            # === BACKWARD: micro-batches 3..0 (reversed) ===
            total_loss = 0.0
            for mb in reversed(range(N_MICROBATCHES)):
                saved_losses[mb].backward()
                total_loss += saved_losses[mb].item()

                # Send input grad back to stage 0 via IB (only local_rank=0)
                grad_to_send = saved_hiddens[mb].grad
                if is_local_rank0:
                    send_buf = grad_to_send.detach().to(torch.bfloat16).contiguous()
                    dist.send(send_buf, dst=stage0_lr0_rank,
                              tag=N_MICROBATCHES + mb)

            return total_loss

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

        optimizer.zero_grad(set_to_none=True)
        avg_loss = gpipe_step()
        optimizer.step()

        step += 1
        if is_rank0 and step % 10 == 0:
            print(f"  step={step:4d}  loss={avg_loss:.4f}  "
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

    # destroy_process_group hangs with composable TP (parallelize_module)
    # due to straggling NCCL collectives during cleanup. Data is already
    # flushed, so just exit.
    import os as _os
    _os.exit(0)


if __name__ == "__main__":
    main()

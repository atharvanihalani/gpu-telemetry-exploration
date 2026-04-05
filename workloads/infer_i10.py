"""
I10 — Multi-Node MoE Inference (TP + EP, torchrun)

Forward-only inference using the same MoE architecture as T14 (TP=8 within
node, EP across nodes). Direct comparison target: T14 training.

Key differences from T14:
  - No backward pass, no optimizer, no gradient allreduce
  - No DDP wrapping (replicate not needed)
  - IB traffic is EP all-to-all only (no periodic DP allreduce)
  - Lower power, lower tensor_sm_ratio expected

Launch (run on BOTH nodes):
    # Node 0:
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
      --master_addr=<NODE0_PRIVATE_IP> --master_port=29500 \
      workloads/infer_i10.py

    # Node 1:
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
      --master_addr=<NODE0_PRIVATE_IP> --master_port=29500 \
      workloads/infer_i10.py

Output (per node):
    data/i10_node{N}_telemetry.csv  (DCGM, 10Hz)
    data/i10_node{N}_ib.csv         (InfiniBand, 10Hz)
    data/i10_node{N}_bmc.csv        (BMC, 2s)
"""

import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

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

# Import config from T1
from workloads.train_t1 import (
    D_MODEL,
    N_LAYERS,
    N_HEADS,
    FFN_MULT,
    SEQ_LEN,
    BATCH_SIZE,
    VOCAB_SIZE,
    WARMUP_S,
)

# ---------------------------------------------------------------------------
# I10-specific config
# ---------------------------------------------------------------------------
DURATION_S = 5 * 60
COOLDOWN_S = 5
N_EXPERTS  = 16      # experts per MoE layer (1 per GPU across BOTH nodes)
TOP_K      = 2       # experts activated per token

OUTPUT_CSV_TEMPLATE = "data/i10_node{}_telemetry.csv"
OUTPUT_IB_TEMPLATE  = "data/i10_node{}_ib.csv"
OUTPUT_BMC_TEMPLATE = "data/i10_node{}_bmc.csv"


# ---------------------------------------------------------------------------
# MoE Layer (from T14/T12, verbatim)
# ---------------------------------------------------------------------------

class MoELayer(nn.Module):
    """Sparse MoE FFN with top-k routing and all-to-all expert parallelism."""

    def __init__(self, d_model, ffn_mult, n_experts, top_k, ep_group):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.ep_group = ep_group
        self.ep_size = dist.get_world_size(group=ep_group)
        self.ep_rank = dist.get_rank(group=ep_group)
        self.d_model = d_model

        # Router: scores each token against each expert
        self.router = nn.Linear(d_model, n_experts, bias=False)

        # Local expert: each GPU hosts 1 expert (n_experts == ep_size)
        self.expert = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model, bias=False),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model, bias=False),
        )

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        N = B * T  # total tokens
        x_flat = x.reshape(N, D)

        # --- Routing ---
        router_logits = self.router(x_flat)           # (N, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # topk_weights: (N, top_k), topk_indices: (N, top_k)

        # Normalize weights so they sum to 1 per token
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # --- Count tokens per expert (for all-to-all sizing) ---
        # Ensure consistent dtype for all-to-all (autocast doesn't cover comms)
        comm_dtype = torch.bfloat16
        x_flat = x_flat.to(comm_dtype)

        # Each token sends top_k copies to different experts
        tokens_expanded = x_flat.unsqueeze(1).expand(-1, self.top_k, -1)  # (N, top_k, D)
        tokens_expanded = tokens_expanded.reshape(N * self.top_k, D)
        expert_ids = topk_indices.reshape(N * self.top_k)                 # (N*top_k,)

        # Sort tokens by expert ID for contiguous all-to-all chunks
        sort_idx = torch.argsort(expert_ids, stable=True)
        tokens_sorted = tokens_expanded[sort_idx]
        expert_ids_sorted = expert_ids[sort_idx]

        # Count how many tokens go to each expert from this GPU
        send_counts = torch.zeros(self.n_experts, dtype=torch.long, device=x.device)
        for e in range(self.n_experts):
            send_counts[e] = (expert_ids_sorted == e).sum()

        # --- All-to-all: exchange token counts ---
        recv_counts = torch.zeros_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.ep_group)

        # --- All-to-all: dispatch tokens ---
        send_splits = send_counts.tolist()
        recv_splits = recv_counts.tolist()

        recv_total = int(recv_counts.sum())
        recv_buf = torch.empty(recv_total, D, dtype=comm_dtype, device=x.device)
        dist.all_to_all_single(
            recv_buf, tokens_sorted,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self.ep_group,
        )

        # --- Expert compute on local tokens ---
        if recv_total > 0:
            expert_out = self.expert(recv_buf)
        else:
            expert_out = recv_buf

        # --- All-to-all: gather results back ---
        send_back_buf = torch.empty(N * self.top_k, D, dtype=comm_dtype, device=x.device)
        dist.all_to_all_single(
            send_back_buf, expert_out,
            output_split_sizes=send_splits,
            input_split_sizes=recv_splits,
            group=self.ep_group,
        )

        # --- Unsort and weighted combine ---
        unsort_idx = torch.argsort(sort_idx)
        tokens_back = send_back_buf[unsort_idx]             # (N*top_k, D)
        tokens_back = tokens_back.reshape(N, self.top_k, D)

        # Weighted sum of top-k expert outputs per token
        output = (tokens_back * topk_weights.unsqueeze(-1)).sum(dim=1)  # (N, D)

        return output.reshape(B, T, D)


# ---------------------------------------------------------------------------
# TP-aware model (attention infers local head count after sharding)
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


# ---------------------------------------------------------------------------
# Transformer blocks: dense (even) and MoE (odd)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Dense block (even indices): attention + dense FFN. Both are TP-sharded."""
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


class MoETransformerBlock(nn.Module):
    """MoE block (odd indices): TP-sharded attention + EP MoE FFN."""
    def __init__(self, d_model, n_heads, ffn_mult, n_experts, top_k, ep_group):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.moe = MoELayer(d_model, ffn_mult, n_experts, top_k, ep_group)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.moe(self.ln2(x))
        return x


class MoEGPT(nn.Module):
    """GPT with alternating dense/MoE blocks (odd = MoE)."""
    def __init__(self, vocab_size, d_model, n_layers, n_heads, ffn_mult,
                 seq_len, n_experts, top_k, ep_group):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)

        blocks = []
        for i in range(n_layers):
            if i % 2 == 1:  # odd blocks are MoE
                blocks.append(MoETransformerBlock(
                    d_model, n_heads, ffn_mult, n_experts, top_k, ep_group
                ))
            else:
                blocks.append(TransformerBlock(d_model, n_heads, ffn_mult))
        self.blocks = nn.ModuleList(blocks)

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
        print(f"I10 — Multi-Node MoE Inference (TP + EP, forward-only)")
        print(f"  world_size={world_size}, nodes={n_nodes}, "
              f"gpus_per_node={nproc_per_node}")
        print(f"  TP={nproc_per_node}-way (attn + dense FFN, NVLink)")
        print(f"  EP={world_size}-way (MoE all-to-all, cross-node over IB)")
        print(f"  NO DP — inference has no gradients to sync")
        print(f"  {N_EXPERTS} experts, top-{TOP_K}, MoE every other block")
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
    # Device Mesh: TP within node only (no DP dimension for inference)
    # ------------------------------------------------------------------
    # For inference we only need TP. We still create a 2D mesh so the
    # rank-to-device mapping is identical to T14, but we won't use the
    # DP dimension for replicate().
    mesh = init_device_mesh(
        "cuda",
        mesh_shape=(n_nodes, nproc_per_node),
        mesh_dim_names=("dp", "tp"),
    )
    tp_mesh = mesh["tp"]

    if is_rank0:
        print(f"  DeviceMesh: dp={n_nodes} (unused), tp={nproc_per_node}")

    # ------------------------------------------------------------------
    # EP group: ONE group spanning ALL 16 GPUs (cross-node)
    # Unlike T14 (intra-node EP), I10 uses cross-node EP so that
    # the all-to-all token shuffle generates IB traffic. Without DP
    # gradient allreduce, this is the only source of inter-node comms.
    # ------------------------------------------------------------------
    all_ranks = list(range(world_size))
    my_ep_group = dist.new_group(all_ranks)

    if is_rank0:
        print(f"  EP group: {all_ranks} (cross-node, all-to-all over IB)")

    # ------------------------------------------------------------------
    # Build model and apply TP (NO DP/replicate — inference only)
    # ------------------------------------------------------------------
    model = MoEGPT(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        ffn_mult=FFN_MULT,
        seq_len=SEQ_LEN,
        n_experts=N_EXPERTS,
        top_k=TOP_K,
        ep_group=my_ep_group,
    ).to(device).to(torch.bfloat16)

    if is_rank0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters (pre-shard): {n_params / 1e9:.2f}B")

    # Apply tensor parallelism (same sharding as T14)
    for i, block in enumerate(model.blocks):
        # Attention: ALWAYS TP-sharded (all blocks)
        parallelize_module(block.attn, tp_mesh, {
            "qkv": ColwiseParallel(),
            "proj": RowwiseParallel(),
        })
        # Dense FFN: only even blocks (odd blocks use MoE, NOT TP-sharded)
        if i % 2 == 0:
            parallelize_module(block.ffn, tp_mesh, {
                "0": ColwiseParallel(),
                "2": RowwiseParallel(),
            })

    # NO replicate() — inference has no gradients to sync across nodes

    if is_rank0:
        local_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters (post-shard, per GPU): {local_params / 1e6:.0f}M")
        print(f"  TP shards attn (all blocks) + dense FFN (even blocks)")
        print(f"  MoE FFN (odd blocks) uses EP, not TP")
        print(f"  No DDP/replicate (forward-only inference)")

    # Set model to eval mode
    model.eval()

    def get_batch():
        """Generate random input batch (same as T14's data gen)."""
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        return x

    # ------------------------------------------------------------------
    # Inference loop (forward-only, no backward/optimizer)
    # ------------------------------------------------------------------
    step     = 0
    t_start  = time.time()
    switched = False

    if is_rank0:
        print(f"  Inference for {DURATION_S}s (warmup={WARMUP_S}s) ...")

    with torch.no_grad():
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

            x = get_batch()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)

            # No loss.backward(), no optimizer.step()
            # Force sync to ensure GPU work completes before timing
            torch.cuda.synchronize()

            step += 1
            if is_rank0 and step % 10 == 0:
                print(f"  step={step:4d}  elapsed={elapsed:.0f}s")

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
        time.sleep(COOLDOWN_S)
        collector.stop()
    if ib_collector:
        ib_collector.stop()
    if bmc_collector:
        bmc_collector.stop()

    if is_rank0:
        print(f"Done. {step} forward passes in {time.time() - t_start:.1f}s")

    # parallelize_module cleanup can hang with straggling NCCL collectives.
    # Data is already flushed, so just _exit (same as T14).
    os._exit(0)


if __name__ == "__main__":
    main()

"""
T12 — MoE Training (Expert Parallelism + DP across nodes)

Mixture-of-Experts transformer with 8 experts per MoE layer, top-2
routing. Every other transformer block uses a sparse MoE FFN instead
of a dense FFN. Expert parallelism (EP) distributes experts across
GPUs within each node; data parallelism (DP) replicates across nodes.

Communication pattern:
  - EP (NVLink): all-to-all token shuffles, twice per MoE layer,
    data-dependent and variable per batch
  - DP (InfiniBand): periodic gradient allreduce every step

This is fundamentally different from DP's periodic heartbeat (T10) and
TP's continuous hum (T11). All-to-all traffic varies per batch based
on which experts are selected by the router.

Launch (run on BOTH nodes within ~60s):

    # Node 0:
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \\
      --master_addr=<NODE0_PRIVATE_IP> --master_port=29500 \\
      workloads/train_t12.py

    # Node 1:
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \\
      --master_addr=<NODE0_PRIVATE_IP> --master_port=29500 \\
      workloads/train_t12.py

Output (per node):
    data/t12_node{N}_telemetry.csv  (DCGM, 10Hz)
    data/t12_node{N}_ib.csv         (InfiniBand, 10Hz)
    data/t12_node{N}_bmc.csv        (BMC, 2s)
"""

import os
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workloads.collect_telemetry import TelemetryCollector
from workloads.collect_ib import IBCollector
from workloads.collect_bmc import BMCCollector

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DURATION_S  = 5 * 60
WARMUP_S    = 30

D_MODEL     = 3072
N_LAYERS    = 28
N_HEADS     = 24
FFN_MULT    = 4
SEQ_LEN     = 2048
BATCH_SIZE  = 4
VOCAB_SIZE  = 32000
LR          = 3e-4

N_EXPERTS   = 8       # experts per MoE layer (1 per GPU within node)
TOP_K       = 2       # experts activated per token

OUTPUT_CSV_TEMPLATE = "data/t12_node{}_telemetry.csv"
OUTPUT_IB_TEMPLATE  = "data/t12_node{}_ib.csv"
OUTPUT_BMC_TEMPLATE = "data/t12_node{}_bmc.csv"


# ---------------------------------------------------------------------------
# MoE Layer
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
# Model
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class TransformerBlock(nn.Module):
    """Standard block with dense FFN."""
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
    """Block with MoE FFN instead of dense FFN."""
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
    """GPT with alternating dense/MoE blocks."""
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
        print(f"T12 — MoE Training (EP + DP)")
        print(f"  world_size={world_size}, nodes={n_nodes}, "
              f"gpus_per_node={nproc_per_node}")
        print(f"  EP={nproc_per_node}-way (within node, NVLink)")
        print(f"  DP={n_nodes}-way (across nodes, InfiniBand)")
        print(f"  {N_EXPERTS} experts, top-{TOP_K}, MoE every other block")
        print(f"  d_model={D_MODEL}, n_layers={N_LAYERS}, n_heads={N_HEADS}")
        print(f"  seq_len={SEQ_LEN}, batch/GPU={BATCH_SIZE}")

    # ------------------------------------------------------------------
    # Telemetry
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
    # Process groups: EP within node, DP across nodes
    # ------------------------------------------------------------------
    # EP groups: GPUs on the same node
    ep_ranks = []
    for n in range(n_nodes):
        start = n * nproc_per_node
        ep_ranks.append(list(range(start, start + nproc_per_node)))

    ep_groups = []
    for ranks in ep_ranks:
        ep_groups.append(dist.new_group(ranks))

    # This rank's EP group
    my_ep_group = ep_groups[node_rank]

    # DP groups: corresponding ranks across nodes
    for local_r in range(nproc_per_node):
        dp_ranks = [n * nproc_per_node + local_r for n in range(n_nodes)]
        dist.new_group(dp_ranks)  # creates the group; DDP finds it automatically

    if is_rank0:
        print(f"  EP groups: {ep_ranks}")

    # ------------------------------------------------------------------
    # Build model
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
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters (all experts on this GPU): {total_params / 1e9:.2f}B")

    # DDP for gradient sync across nodes (handles dense + expert params)
    model = DDP(model, device_ids=[local_rank])

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
        loss.backward()
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

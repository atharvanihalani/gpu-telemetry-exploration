"""
T5 — Gradient Checkpointing Pre-Training (DDP, synthetic data)

Fork of T1 with gradient checkpointing enabled via torch.utils.checkpoint.
Trades compute for memory: intermediate activations are discarded during the
forward pass and recomputed during backward. This allows fitting a larger
model (6.4B params) that OOMed in T1 without checkpointing.

Evasion angle: memory footprint drops significantly because activations are
not stored. With the 6.4B config, GPU memory (~35-40 GB) overlaps with what
a large inference model might use. A detector relying solely on "high memory
= training" would be fooled. However, power, SM utilization, and NVLink
allreduce heartbeat remain strong training indicators.

Primary config: 6.4B params (d_model=4096, 32 layers, 32 heads)
  - OOMed in T1 without checkpointing
  - Should fit with gradient checkpointing (~35-40 GB)

Fallback config: 3.2B params (d_model=3072, 28 layers, 24 heads)
  - Same as T1, used if 6.4B still OOMs for any reason

The script tries 6.4B first, runs a test forward+backward pass to trigger
any OOM early, and automatically falls back to 3.2B with a clear log message.

Launch:
    torchrun --nproc_per_node=8 workloads/train_t5.py

Config (edit at top of file):
    DURATION_S      total wall-clock training time (default 5 min)
    WARMUP_S        initial phase excluded from "steady" analysis
    SEQ_LEN / BATCH_SIZE          sequence and batch dimensions
"""

import os
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workloads.collect_telemetry import TelemetryCollector

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DURATION_S = 5 * 60   # total run time
WARMUP_S   = 30       # first N seconds labelled "warmup", excluded from analysis

# Primary config (~6.4B params) — fits with gradient checkpointing
# OOMed in T1 without checkpointing: activations at batch=4, seq=2048
# consumed ~75GB leaving no room for AdamW states.
D_MODEL_LARGE  = 4096
N_LAYERS_LARGE = 32
N_HEADS_LARGE  = 32

# Fallback config (~3.2B params) — same as T1
D_MODEL_SMALL  = 3072
N_LAYERS_SMALL = 28
N_HEADS_SMALL  = 24

FFN_MULT   = 4        # FFN hidden dim = FFN_MULT * D_MODEL
SEQ_LEN    = 2048
BATCH_SIZE = 4         # per-GPU; reduce if OOM
VOCAB_SIZE = 32000
LR         = 3e-4

OUTPUT_CSV = "data/t5_telemetry.csv"

# ---------------------------------------------------------------------------
# Model (same architecture as T1, but forward uses gradient checkpointing)
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
        q, k, v = qkv.unbind(2)                  # (B, T, H, D)
        q = q.transpose(1, 2)                     # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )
        y = y.transpose(1, 2).reshape(B, T, C)
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
    """GPT with gradient checkpointing on every transformer block."""

    def __init__(self, vocab_size, d_model, n_layers, n_heads, ffn_mult, seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks  = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_mult) for _ in range(n_layers)
        ])
        self.ln_f    = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device).unsqueeze(0)
        x    = self.tok_emb(idx) + self.pos_emb(pos)
        # Gradient checkpointing: discard activations during forward,
        # recompute them during backward. Saves ~60% activation memory
        # at the cost of ~33% more compute per step.
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False)
        x = self.ln_f(x)
        return self.head(x)


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
        print(f"T5 — Gradient Checkpointing Pre-Training")
        print(f"DDP world_size={world_size}")
        print(f"Sequence length={SEQ_LEN}, batch/GPU={BATCH_SIZE}")

    # Telemetry — only rank 0 collects (pynvml sees all GPUs)
    collector = None
    if is_rank0:
        collector = TelemetryCollector(OUTPUT_CSV)
        collector.start()
        collector.set_phase("warmup")

    # ------------------------------------------------------------------
    # Model setup with automatic fallback
    # Try 6.4B first; if OOM, fall back to 3.2B
    # ------------------------------------------------------------------
    config_name = None
    d_model = None
    n_layers = None
    n_heads = None

    try:
        if is_rank0:
            print(f"Attempting 6.4B config: d_model={D_MODEL_LARGE}, "
                  f"n_layers={N_LAYERS_LARGE}, n_heads={N_HEADS_LARGE}")

        model = GPT(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL_LARGE,
            n_layers=N_LAYERS_LARGE,
            n_heads=N_HEADS_LARGE,
            ffn_mult=FFN_MULT,
            seq_len=SEQ_LEN,
        ).to(device).to(torch.bfloat16)

        model = DDP(model, device_ids=[local_rank])
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

        # Test forward + backward to trigger any OOM early
        x_test = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(x_test)
            loss = out.sum()
        loss.backward()
        optimizer.zero_grad(set_to_none=True)
        del x_test, out, loss
        torch.cuda.empty_cache()

        config_name = "6.4B"
        d_model = D_MODEL_LARGE
        n_layers = N_LAYERS_LARGE
        n_heads = N_HEADS_LARGE

        if is_rank0:
            print("Using 6.4B config (gradient checkpointing enabled)")

    except torch.cuda.OutOfMemoryError:
        # Clean up failed attempt
        try:
            del model, optimizer
        except NameError:
            pass
        torch.cuda.empty_cache()

        if is_rank0:
            print("6.4B OOM — falling back to 3.2B config")
            print(f"Fallback config: d_model={D_MODEL_SMALL}, "
                  f"n_layers={N_LAYERS_SMALL}, n_heads={N_HEADS_SMALL}")

        model = GPT(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL_SMALL,
            n_layers=N_LAYERS_SMALL,
            n_heads=N_HEADS_SMALL,
            ffn_mult=FFN_MULT,
            seq_len=SEQ_LEN,
        ).to(device).to(torch.bfloat16)

        model = DDP(model, device_ids=[local_rank])
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

        config_name = "3.2B"
        d_model = D_MODEL_SMALL
        n_layers = N_LAYERS_SMALL
        n_heads = N_HEADS_SMALL

        if is_rank0:
            print("Using 3.2B fallback config (gradient checkpointing enabled)")

    # Count and report params
    if is_rank0:
        n_params = sum(p.numel() for p in model.module.parameters())
        print(f"Config: {config_name} | d_model={d_model} | "
              f"n_layers={n_layers} | n_heads={n_heads}")
        print(f"Parameters: {n_params / 1e9:.2f}B")

    loss_fn = nn.CrossEntropyLoss()

    # Synthetic data — random token IDs
    def get_batch():
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        return x, y

    # Training loop
    step     = 0
    t_start  = time.time()
    switched = False

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
        print(f"Done. {step} steps in {time.time() - t_start:.1f}s "
              f"({config_name} config with gradient checkpointing)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

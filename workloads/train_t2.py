"""
T2 — Small Pre-Training (DDP, synthetic data)

Tiny-model variant of T1. Same DDP training loop, same synthetic data, same
duration — but with a ~125M-param model instead of T1's ~3.4B.

The goal: test whether the allreduce heartbeat pattern (periodic, synchronized
NVLink bursts) is still detectable when power draw and memory usage drop to
levels that overlap with inference workloads. If the temporal pattern persists
even at low absolute magnitudes, it's a more robust training signal than any
single-metric threshold.

Model config:
    d_model=768, n_layers=12, n_heads=12, ffn_mult=4  (~125M params)
    Compared to T1: d_model=3072, n_layers=28, n_heads=24 (~3.4B params)

Everything else is identical to T1:
    - DDP across 8 GPUs (torchrun --nproc_per_node=8)
    - Synthetic random-token data
    - bf16 AMP, AdamW optimizer
    - 5 min total (30s warmup, ~270s steady, 5s cooldown)
    - 1 Hz pynvml telemetry via shared collector

Expected telemetry vs T1:
    - Power: significantly lower (~150-250W vs ~400W) — not fully saturating GPU
    - SM util: still high during compute, but steps are much faster
    - Memory: very low (~2-3 GB vs ~67 GB) — close to inference levels
    - NVLink: same allreduce heartbeat but at higher frequency (faster steps)

Launch:
    torchrun --nproc_per_node=8 workloads/train_t2.py

Output:
    data/t2_telemetry.csv
"""

import os
import sys
import time
import math

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workloads.collect_telemetry import TelemetryCollector

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DURATION_S  = 5 * 60   # total run time
WARMUP_S    = 30       # first N seconds labelled "warmup", excluded from analysis

# Small config (~125M params, ~2-3GB GPU mem with AdamW)
# Deliberately tiny to test whether temporal patterns (allreduce heartbeat)
# persist when magnitude signals (power, memory) drop to inference-like levels.
D_MODEL     = 768
N_LAYERS    = 12
N_HEADS     = 12
FFN_MULT    = 4        # FFN hidden dim = FFN_MULT * D_MODEL

SEQ_LEN     = 2048
BATCH_SIZE  = 4        # per-GPU; same as T1 to isolate model-size variable

VOCAB_SIZE  = 32000
LR          = 3e-4

OUTPUT_CSV  = "data/t2_telemetry.csv"

# ---------------------------------------------------------------------------
# Model (same architecture as T1, just smaller config)
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
        # Flash attention if available, else manual
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
        for block in self.blocks:
            x = block(x)
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
        print(f"[T2] DDP world_size={world_size}, d_model={D_MODEL}, "
              f"n_layers={N_LAYERS}, n_heads={N_HEADS}")
        print(f"[T2] Sequence length={SEQ_LEN}, batch/GPU={BATCH_SIZE}")

    # Telemetry — only rank 0 collects (pynvml sees all GPUs)
    collector = None
    if is_rank0:
        collector = TelemetryCollector(OUTPUT_CSV)
        collector.start()
        collector.set_phase("warmup")

    # Build model
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
        print(f"[T2] Parameters: {n_params / 1e6:.1f}M")

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
        print(f"[T2] Training for {DURATION_S}s (warmup={WARMUP_S}s) ...")

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
            # logits: (B, T, V) -> (B*T, V); targets: (B*T,)
            loss = loss_fn(
                logits.reshape(-1, VOCAB_SIZE),
                y.reshape(-1),
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()          # allreduce happens here (DDP hook)
        optimizer.step()

        step += 1
        if is_rank0 and step % 50 == 0:
            print(f"  step={step:4d}  loss={loss.item():.4f}  "
                  f"elapsed={elapsed:.0f}s")

    if is_rank0:
        if collector:
            collector.set_phase("cooldown")
            time.sleep(5)
            collector.stop()
        print(f"[T2] Done. {step} steps in {time.time() - t_start:.1f}s")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

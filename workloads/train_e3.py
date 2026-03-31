"""
E3 — Intermittent Training (Duty-Cycled)

Evasion workload: trains for TRAIN_WINDOW_S seconds, then sleeps for
SLEEP_WINDOW_S seconds, repeating until DURATION_S is reached. Breaks the
"sustained flat power" assumption that distinguishes training from inference.

Fork of T1 (train_t1.py) with a duty cycle wrapper around the training loop.
Same model (3.2B GPT), same DDP setup, same synthetic data.

Expected telemetry:
    - Power: square wave — ~400W during training bursts, ~60-80W during sleep
    - SM util: 100% during bursts, 0% during sleep
    - Memory: stays high throughout (model + optimizer states remain allocated)
    - NVLink: allreduce heartbeat during bursts, zero during sleep

Launch:
    torchrun --nproc_per_node=8 workloads/train_e3.py

Config (edit at top of file):
    DURATION_S          total wall-clock time (default 5 min)
    WARMUP_S            continuous training before duty cycling begins
    TRAIN_WINDOW_S      seconds of training per duty cycle
    SLEEP_WINDOW_S      seconds of idle per duty cycle
    D_MODEL / N_LAYERS / N_HEADS  model architecture (same as T1)
    SEQ_LEN / BATCH_SIZE          sequence and batch dimensions
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
DURATION_S      = 5 * 60   # total run time
WARMUP_S        = 30       # continuous training before duty cycling starts

# Duty cycle parameters
TRAIN_WINDOW_S  = 30       # train for this long per cycle
SLEEP_WINDOW_S  = 10       # then idle for this long
# Duty cycle = 30/40 = 75%

# Model — same 3.2B GPT as T1
D_MODEL     = 3072
N_LAYERS    = 28
N_HEADS     = 24
FFN_MULT    = 4        # FFN hidden dim = FFN_MULT * D_MODEL

SEQ_LEN     = 2048
BATCH_SIZE  = 4        # per-GPU; reduce if OOM

VOCAB_SIZE  = 32000
LR          = 3e-4

OUTPUT_CSV  = "data/e3_telemetry.csv"

# ---------------------------------------------------------------------------
# Model (identical to T1)
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
        duty_pct = TRAIN_WINDOW_S / (TRAIN_WINDOW_S + SLEEP_WINDOW_S) * 100
        print(f"E3 — Intermittent Training (Duty-Cycled)")
        print(f"DDP world_size={world_size}, d_model={D_MODEL}, "
              f"n_layers={N_LAYERS}, n_heads={N_HEADS}")
        print(f"Sequence length={SEQ_LEN}, batch/GPU={BATCH_SIZE}")
        print(f"Duty cycle: {TRAIN_WINDOW_S}s train / {SLEEP_WINDOW_S}s sleep "
              f"= {duty_pct:.0f}%")

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
        print(f"Parameters: {n_params / 1e9:.2f}B")

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

    step      = 0
    cycle     = 0
    t_start   = time.time()

    if is_rank0:
        print(f"Training for {DURATION_S}s (warmup={WARMUP_S}s continuous, "
              f"then duty cycling) ...")

    # ------------------------------------------------------------------
    # Phase 1: Warmup — continuous training, no duty cycling
    # ------------------------------------------------------------------
    while True:
        elapsed = time.time() - t_start
        if elapsed >= WARMUP_S or elapsed >= DURATION_S:
            break

        x, y = get_batch()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        step += 1
        if is_rank0 and step % 10 == 0:
            print(f"  [warmup] step={step:4d}  loss={loss.item():.4f}  "
                  f"elapsed={elapsed:.0f}s")

    if is_rank0:
        print(f"Warmup complete ({step} steps). Starting duty cycling.")

    # ------------------------------------------------------------------
    # Phase 2: Duty cycling — train for TRAIN_WINDOW_S, sleep SLEEP_WINDOW_S
    # ------------------------------------------------------------------
    # Set initial phase to "steady" for the first training burst
    if is_rank0 and collector:
        collector.set_phase("steady")

    while True:
        elapsed = time.time() - t_start
        if elapsed >= DURATION_S:
            break

        # --- Training burst ---
        cycle += 1
        burst_start = time.time()
        if is_rank0:
            print(f"  [cycle {cycle}] training burst start "
                  f"(elapsed={elapsed:.0f}s)")

        while time.time() - burst_start < TRAIN_WINDOW_S:
            if time.time() - t_start >= DURATION_S:
                break

            x, y = get_batch()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1
            if is_rank0 and step % 10 == 0:
                print(f"  [cycle {cycle}] step={step:4d}  "
                      f"loss={loss.item():.4f}  "
                      f"elapsed={time.time() - t_start:.0f}s")

        # Check if we've exceeded total duration
        if time.time() - t_start >= DURATION_S:
            break

        # --- Sleep window ---
        dist.barrier()  # sync all GPUs before sleeping
        if is_rank0 and collector:
            collector.set_phase("sleep")
        if is_rank0:
            print(f"  [cycle {cycle}] sleeping {SLEEP_WINDOW_S}s "
                  f"(elapsed={time.time() - t_start:.0f}s)")

        time.sleep(SLEEP_WINDOW_S)

        dist.barrier()  # sync all GPUs after sleeping
        if is_rank0 and collector:
            collector.set_phase("steady")

    # ------------------------------------------------------------------
    # Cooldown
    # ------------------------------------------------------------------
    if is_rank0:
        if collector:
            collector.set_phase("cooldown")
            time.sleep(5)
            collector.stop()
        print(f"Done. {step} steps, {cycle} duty cycles in "
              f"{time.time() - t_start:.1f}s")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""
T3 — Gradient Accumulation Training (DDP, synthetic data)

Fork of T1 with gradient accumulation. Accumulates ACCUM_STEPS microbatch
gradients before running allreduce, stretching the allreduce heartbeat period
by ACCUM_STEPS×. This tests whether reducing communication frequency can evade
temporal pattern detectors tuned to T1's ~15–20s step period.

Key differences from T1:
  - model.no_sync() suppresses allreduce on all microsteps except the last
  - Loss is divided by ACCUM_STEPS so effective gradient magnitude matches T1
  - optimizer.step() / zero_grad() happen only after all ACCUM_STEPS microbatches
  - Step logging counts "optimizer steps" (1 logged step = ACCUM_STEPS microbatches)

Expected telemetry vs T1:
  - Power: same sustained ~400W (GPU is always computing)
  - SM util: ~100% (same as T1)
  - Memory: same as T1 (gradients accumulate in-place via +=)
  - NVLink: bursts spaced ACCUM_STEPS× further apart
    With ACCUM_STEPS=16, heartbeat period goes from ~15–20s to ~4–5 min

Launch:
    torchrun --nproc_per_node=8 workloads/train_t3.py

Config (edit at top of file):
    ACCUM_STEPS     gradient accumulation steps (default 16)
    DURATION_S      total wall-clock training time (default 5 min)
    WARMUP_S        initial phase excluded from "steady" analysis
    D_MODEL / N_LAYERS / N_HEADS  model architecture
    SEQ_LEN / BATCH_SIZE          sequence and batch dimensions
"""

import os
import sys
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workloads.collect_telemetry import TelemetryCollector

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ACCUM_STEPS = 16       # gradient accumulation steps — higher = less frequent allreduce

DURATION_S  = 5 * 60   # total run time
WARMUP_S    = 30        # first N seconds labelled "warmup", excluded from analysis

# Medium config (~3.2B params, ~38GB GPU mem with AdamW)
# Large config (d_model=4096, n_layers=32, n_heads=32) OOMs on 80GB GPUs:
# activations at batch=4, seq=2048 consume ~75GB leaving no room for AdamW states.
D_MODEL     = 3072
N_LAYERS    = 28
N_HEADS     = 24
FFN_MULT    = 4        # FFN hidden dim = FFN_MULT * D_MODEL

SEQ_LEN     = 2048
BATCH_SIZE  = 4        # per-GPU; reduce if OOM

VOCAB_SIZE  = 32000
LR          = 3e-4

OUTPUT_CSV  = "data/t3_telemetry.csv"

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
        print(f"T3 gradient accumulation: ACCUM_STEPS={ACCUM_STEPS}")
        print(f"DDP world_size={world_size}, d_model={D_MODEL}, "
              f"n_layers={N_LAYERS}, n_heads={N_HEADS}")
        print(f"Sequence length={SEQ_LEN}, batch/GPU={BATCH_SIZE}")
        print(f"Effective batch/GPU = {BATCH_SIZE * ACCUM_STEPS} "
              f"({BATCH_SIZE} × {ACCUM_STEPS} accum steps)")

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

    # Training loop with gradient accumulation
    step           = 0   # optimizer steps (each = ACCUM_STEPS microbatches)
    micro_count    = 0   # total microbatches processed
    t_start        = time.time()
    switched       = False

    if is_rank0:
        print(f"Training for {DURATION_S}s (warmup={WARMUP_S}s) ...")
        print(f"Allreduce every {ACCUM_STEPS} microbatches "
              f"(~{ACCUM_STEPS}× longer heartbeat period than T1)")

    while True:
        elapsed = time.time() - t_start
        if elapsed >= DURATION_S:
            break

        # Switch phase label after warmup
        if not switched and elapsed >= WARMUP_S:
            if is_rank0 and collector:
                collector.set_phase("steady")
            switched = True

        # --- Gradient accumulation loop ---
        for micro_step in range(ACCUM_STEPS):
            # Check time limit inside inner loop to avoid overrunning
            if time.time() - t_start >= DURATION_S:
                break

            x, y = get_batch()

            # Suppress allreduce on all but the last microstep
            ctx = model.no_sync() if micro_step < ACCUM_STEPS - 1 else nullcontext()
            with ctx:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x)
                    loss = loss_fn(
                        logits.reshape(-1, VOCAB_SIZE),
                        y.reshape(-1),
                    )
                    loss = loss / ACCUM_STEPS  # scale loss for accumulation
                loss.backward()  # allreduce only on last microstep

            micro_count += 1

        # Optimizer step after accumulating all microbatches
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        step += 1
        if is_rank0 and step % 5 == 0:
            elapsed = time.time() - t_start
            print(f"  opt_step={step:4d}  microbatches={micro_count:5d}  "
                  f"loss={loss.item() * ACCUM_STEPS:.4f}  "
                  f"elapsed={elapsed:.0f}s")

    if is_rank0:
        if collector:
            collector.set_phase("cooldown")
            time.sleep(5)
            collector.stop()
        print(f"Done. {step} optimizer steps ({micro_count} microbatches) "
              f"in {time.time() - t_start:.1f}s")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

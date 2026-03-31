"""
T6 — ZeRO-3 / CPU Offloading (FSDP, synthetic data)

Tests whether CPU offloading can eliminate the "high GPU memory" training
signal. Uses FSDP with CPUOffload(offload_params=True) so parameters live
on CPU and are only streamed to GPU for the active layer. Optimizer states
are also CPU-resident (consequence of CPU-offloaded params).

Expected telemetry vs T1 (DDP):
  - GPU memory: dramatically lower (only active layer on GPU at a time)
  - Power: lower (GPU idles while waiting for CPU↔GPU transfers)
  - SM util: bursty (compute bursts interleaved with PCIe stalls)
  - PCIe traffic: very high (constant param/grad streaming)
  - NVLink: still active (FSDP gathers shards across GPUs) but different
    pattern from DDP allreduce

Launch:
    torchrun --nproc_per_node=8 workloads/train_t6.py

Config (edit at top of file):
    DURATION_S      total wall-clock training time (default 5 min)
    WARMUP_S        initial phase excluded from "steady" analysis
    D_MODEL / N_LAYERS / N_HEADS  model architecture (same as T1)
    SEQ_LEN / BATCH_SIZE          sequence and batch dimensions
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
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workloads.collect_telemetry import TelemetryCollector

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DURATION_S  = 5 * 60   # total run time
WARMUP_S    = 30        # first N seconds labelled "warmup"

# Same architecture as T1 (~3.2B params)
D_MODEL     = 3072
N_LAYERS    = 28
N_HEADS     = 24
FFN_MULT    = 4         # FFN hidden dim = FFN_MULT * D_MODEL

SEQ_LEN     = 2048
BATCH_SIZE  = 4          # per-GPU; may be able to increase with CPU offload

VOCAB_SIZE  = 32000
LR          = 3e-4

OUTPUT_CSV  = "data/t6_telemetry.csv"

# ---------------------------------------------------------------------------
# Model  (identical to T1)
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
# Helpers
# ---------------------------------------------------------------------------

def get_cpu_ram_gb():
    """Return total system RAM in GB using os.sysconf (no psutil needed)."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (pages * page_size) / (1024 ** 3)
    except (ValueError, OSError):
        return -1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Distributed init
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    is_rank0 = rank == 0

    if is_rank0:
        cpu_ram = get_cpu_ram_gb()
        print(f"T6 — FSDP + CPU Offload")
        print(f"  world_size={world_size}, d_model={D_MODEL}, "
              f"n_layers={N_LAYERS}, n_heads={N_HEADS}")
        print(f"  seq_len={SEQ_LEN}, batch/GPU={BATCH_SIZE}")
        print(f"  CPU RAM: {cpu_ram:.0f} GB")

    # Telemetry — only rank 0 collects (pynvml sees all GPUs)
    collector = None
    if is_rank0:
        collector = TelemetryCollector(OUTPUT_CSV)
        collector.start()
        collector.set_phase("warmup")

    # Build model on CPU first (FSDP with CPU offload expects this)
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

    # -----------------------------------------------------------------------
    # FSDP wrapping with CPU offload
    # -----------------------------------------------------------------------

    # Auto-wrap: each TransformerBlock becomes its own FSDP unit.
    # This ensures only one block's params are on GPU at a time.
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    # Mixed precision: bf16 for compute, bf16 for allreduce, bf16 for buffers
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # Try FSDP with CPU offload first; fall back to GPU-only FSDP if it fails
    use_cpu_offload = True
    try:
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            cpu_offload=CPUOffload(offload_params=True),
            mixed_precision=mp_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3 equivalent
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=local_rank,
            sync_module_states=True,  # broadcast rank-0 weights to all ranks
            use_orig_params=False,
        )
        if is_rank0:
            print("  FSDP mode: FULL_SHARD + CPUOffload(offload_params=True)")
    except Exception as e:
        if is_rank0:
            print(f"  CPU offload failed ({e}), falling back to GPU-only FSDP")
        use_cpu_offload = False
        # Rebuild model (previous one may be in a bad state)
        model = GPT(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            n_layers=N_LAYERS,
            n_heads=N_HEADS,
            ffn_mult=FFN_MULT,
            seq_len=SEQ_LEN,
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
            print("  FSDP mode: FULL_SHARD (GPU-only, no CPU offload)")

    # Optimizer — with CPU offload, FSDP automatically places optimizer
    # states on CPU alongside the parameters.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn   = nn.CrossEntropyLoss()

    if is_rank0:
        print(f"  Offload mode: {'CPU' if use_cpu_offload else 'GPU-only'}")
        # Report GPU memory after model setup
        torch.cuda.synchronize()
        mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
        mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        print(f"  GPU memory after setup: {mem_alloc:.2f} GB allocated, "
              f"{mem_reserved:.2f} GB reserved")

    # Synthetic data — random token IDs
    def get_batch():
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        return x, y

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    step      = 0
    t_start   = time.time()
    switched  = False

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

        # Forward — FSDP handles moving params to GPU on demand
        logits = model(x)
        loss = loss_fn(
            logits.reshape(-1, VOCAB_SIZE),
            y.reshape(-1),
        )

        # Backward — FSDP handles gradient reduce-scatter + offload
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        step += 1
        if is_rank0 and step % 5 == 0:
            mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
            print(f"  step={step:4d}  loss={loss.item():.4f}  "
                  f"elapsed={elapsed:.0f}s  gpu_mem={mem_alloc:.2f}GB")

    # Cooldown
    if is_rank0:
        if collector:
            collector.set_phase("cooldown")
            time.sleep(5)
            collector.stop()
        print(f"Done. {step} steps in {time.time() - t_start:.1f}s")
        print(f"  Offload mode used: {'CPU' if use_cpu_offload else 'GPU-only'}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

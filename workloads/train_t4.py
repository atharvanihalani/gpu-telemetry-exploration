"""
T4 — Pipeline Parallelism Training (GPipe schedule, synthetic data)

Splits the GPT model across 8 GPUs by layer (pipeline parallelism) instead
of replicating it on every GPU (data parallelism as in T1). This produces a
fundamentally different inter-GPU communication pattern:

  - T1 (DDP): all-to-all allreduce every backward pass — the "heartbeat"
  - T4 (pipeline): sequential point-to-point activation/gradient passing
    between adjacent stages (GPU0->1->...->7 forward, GPU7->...->0 backward)

The allreduce heartbeat is the strongest single training signal in T1.
Pipeline parallelism eliminates it entirely, replacing it with P2P traffic
that looks more like tensor-parallel inference. This tests whether detectors
that rely on the allreduce signature can be evaded.

Expected telemetry vs T1:
  - NVLink: P2P between adjacent stages only, NOT all-to-all
  - Power: "wave" pattern (pipeline bubble = idle stages), not flat high
  - SM util: lower average (~50-70%) due to pipeline bubble overhead
  - Memory: lower per GPU (each holds only ~3-4 transformer blocks)

Uses PyTorch's torch.distributed.pipelining (PiPPy) with ScheduleGPipe.

Launch:
    torchrun --nproc_per_node=8 workloads/train_t4.py

Config (edit at top of file):
    DURATION_S      total wall-clock training time (default 5 min)
    WARMUP_S        initial phase excluded from "steady" analysis
    D_MODEL / N_LAYERS / N_HEADS  model architecture (same as T1)
    SEQ_LEN / BATCH_SIZE          sequence and batch dimensions
    N_MICROBATCHES  number of microbatches to fill the pipeline
"""

import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn

from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleGPipe

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workloads.collect_telemetry import TelemetryCollector

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DURATION_S     = 5 * 60   # total run time
WARMUP_S       = 30       # first N seconds labelled "warmup"

# Model architecture — same as T1 (~3.2B params total, split across 8 GPUs)
D_MODEL        = 3072
N_LAYERS       = 28
N_HEADS        = 24
FFN_MULT       = 4        # FFN hidden dim = FFN_MULT * D_MODEL

SEQ_LEN        = 2048
BATCH_SIZE     = 8         # total batch size (chunked into N_MICROBATCHES microbatches)
N_MICROBATCHES = 8         # GPipe microbatch count — fills the pipeline
                           # Each microbatch = BATCH_SIZE // N_MICROBATCHES = 1 sample
                           # Increase BATCH_SIZE for larger microbatches (watch for OOM)

VOCAB_SIZE     = 32000
LR             = 3e-4

OUTPUT_CSV     = "data/t4_telemetry.csv"

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
# Pipeline split specification
# ---------------------------------------------------------------------------
# 28 layers across 8 GPUs. Split points mark the BEGINNING of a new stage.
#
# Stage 0 (GPU 0): embedding + blocks 0-2       (3 blocks + embed)
# Stage 1 (GPU 1): blocks 3-6                   (4 blocks)
# Stage 2 (GPU 2): blocks 7-9                   (3 blocks)
# Stage 3 (GPU 3): blocks 10-13                 (4 blocks)
# Stage 4 (GPU 4): blocks 14-16                 (3 blocks)
# Stage 5 (GPU 5): blocks 17-20                 (4 blocks)
# Stage 6 (GPU 6): blocks 21-23                 (3 blocks)
# Stage 7 (GPU 7): blocks 24-27 + ln_f + head   (4 blocks + head)

SPLIT_SPEC = {
    "blocks.3":  SplitPoint.BEGINNING,   # stage 1 starts here
    "blocks.7":  SplitPoint.BEGINNING,   # stage 2
    "blocks.10": SplitPoint.BEGINNING,   # stage 3
    "blocks.14": SplitPoint.BEGINNING,   # stage 4
    "blocks.17": SplitPoint.BEGINNING,   # stage 5
    "blocks.21": SplitPoint.BEGINNING,   # stage 6
    "blocks.24": SplitPoint.BEGINNING,   # stage 7
}


# ---------------------------------------------------------------------------
# Loss function (must match T1: token-wise cross-entropy)
# ---------------------------------------------------------------------------
def tokenwise_loss_fn(logits, targets):
    """Cross-entropy loss reshaped for (B, T, V) logits and (B, T) targets."""
    return nn.functional.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE),
        targets.reshape(-1),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Process group init (same as T1) ---
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    is_rank0 = rank == 0
    is_last  = rank == world_size - 1

    if is_rank0:
        print(f"Pipeline world_size={world_size}, d_model={D_MODEL}, "
              f"n_layers={N_LAYERS}, n_heads={N_HEADS}")
        print(f"Sequence length={SEQ_LEN}, batch={BATCH_SIZE}, "
              f"microbatches={N_MICROBATCHES}")

    # --- Telemetry (rank 0 only, sees all GPUs via pynvml) ---
    collector = None
    if is_rank0:
        collector = TelemetryCollector(OUTPUT_CSV)
        collector.start()
        collector.set_phase("warmup")

    # --- Build full model on CPU for tracing ---
    # PiPPy traces the model with example inputs to determine the graph
    # and split it into stages. Only the stage assigned to this rank will
    # be moved to GPU, so the full model on CPU is temporary.
    if is_rank0:
        print("Building model on CPU for pipeline tracing...")

    model_cpu = GPT(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        ffn_mult=FFN_MULT,
        seq_len=SEQ_LEN,
    )

    # Cast to bf16 before tracing so PiPPy records bf16 activation dtypes.
    # This must match the autocast dtype used in the training loop below.
    # LayerNorm weight/bias stay fp32 internally but outputs are bf16 under
    # autocast — casting the full model here keeps tracing consistent.
    model_cpu = model_cpu.to(torch.bfloat16)

    if is_rank0:
        n_params = sum(p.numel() for p in model_cpu.parameters())
        print(f"Total parameters: {n_params / 1e9:.2f}B")

    # --- Trace and split the model ---
    # pipeline() needs an example *microbatch* input for tracing.
    # ScheduleGPipe.step() will split the full batch into N_MICROBATCHES chunks,
    # so the microbatch size = BATCH_SIZE // N_MICROBATCHES.
    mb_size = max(1, BATCH_SIZE // N_MICROBATCHES)
    example_input = torch.randint(0, VOCAB_SIZE, (mb_size, SEQ_LEN))

    if is_rank0:
        print("Tracing model with PiPPy pipeline()...")

    pipe = pipeline(
        module=model_cpu,
        mb_args=(example_input,),
        split_spec=SPLIT_SPEC,
    )

    assert pipe.num_stages == world_size, (
        f"Pipeline has {pipe.num_stages} stages but world_size={world_size}. "
        f"Adjust SPLIT_SPEC to produce exactly {world_size} stages."
    )

    if is_rank0:
        print(f"Pipeline split into {pipe.num_stages} stages:")
        for i in range(pipe.num_stages):
            stage_mod = pipe.get_stage_module(i)
            n_p = sum(p.numel() for p in stage_mod.parameters())
            print(f"  Stage {i}: {n_p / 1e6:.1f}M params")

    # --- Build this rank's pipeline stage ---
    # build_stage() extracts this rank's submodule and moves it to device.
    # It also runs a forward pass to determine activation shapes for P2P buffers.
    stage = pipe.build_stage(rank, device=device)

    # Get a reference to this rank's submodule (for optimizer and param counting)
    stage_mod = pipe.get_stage_module(rank)

    if is_rank0:
        local_params = sum(p.numel() for p in stage_mod.parameters())
        print(f"Rank {rank} local params: {local_params / 1e6:.1f}M")

    # --- Optimizer (per-stage, each rank optimizes only its own layers) ---
    optimizer = torch.optim.AdamW(stage_mod.parameters(), lr=LR)

    # --- Schedule ---
    # ScheduleGPipe: all microbatches do forward, then all do backward (fill-drain).
    # Loss function is only evaluated on the last stage.
    schedule = ScheduleGPipe(
        stage=stage,
        n_microbatches=N_MICROBATCHES,
        loss_fn=tokenwise_loss_fn,
    )

    # --- Training loop ---
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

        # GPipe schedule handles microbatching and P2P communication internally.
        # - Rank 0 (first stage): provides input tokens
        # - Last rank (last stage): provides target tokens for loss computation
        # - Middle ranks: call step() with no args, receive/send activations via P2P
        #
        # Synthetic random data — loss value is meaningless, but the compute
        # pattern (forward + backward + optimizer) matches real training exactly.
        losses = []

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if is_rank0:
                x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
                y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
                schedule.step(x, target=y, losses=losses)
            elif is_last:
                y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
                schedule.step(target=y, losses=losses)
            else:
                schedule.step()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        step += 1
        if is_rank0 and step % 10 == 0:
            loss_val = losses[0].item() if losses else float("nan")
            print(f"  step={step:4d}  loss={loss_val:.4f}  "
                  f"elapsed={elapsed:.0f}s")

    # --- Cleanup ---
    if is_rank0:
        if collector:
            collector.set_phase("cooldown")
            time.sleep(5)
            collector.stop()
        print(f"Done. {step} steps in {time.time() - t_start:.1f}s")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

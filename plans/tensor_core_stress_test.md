# Plan: Tensor Core Signal Stress Test — Training vs Prefill Inference

## Context

We want to test whether `tensor_sm_ratio` (tensor_active / sm_active) can distinguish training from **prefill-only inference** — the hardest case for our classifier, because prefill is just forward passes through big matmuls, which looks very similar to training's forward pass.

Real frontier labs run disaggregated prefill/decode. Prefill servers maximize tensor core utilization with continuous batched forward passes. If tensor_sm_ratio can't distinguish this from training, we need other signals.

## Hardware

- 1x H100 NVL node (Vast.ai), 8 GPUs, 96GB each, 400W TDP
- NVLink only between pairs (NV12: 0-1, 2-3, 4-5, 6-7), no NVSwitch
- DCGM running, single-node only (no IB/BMC needed)

---

## Workload A: Training (MoE)

**Script**: New `workloads/train_t20.py` — Mixtral 8x7B fine-tune

- Load `mistralai/Mixtral-8x7B-v0.1` (public, no HF_TOKEN needed, ~88GB bf16)
- `device_map='auto'` distributes layers across 8 GPUs
- Fine-tune with synthetic random tokens (consistent with existing workloads)
- bf16 autocast, AdamW optimizer
- Fixed batch size, fixed sequence length (2048 tokens)
- 5 min run: 30s warmup → 270s steady → 5s cooldown
- Telemetry: `TelemetryCollector` on rank 0 → `data/t20_telemetry.csv`

**Why Mixtral for training too?** Same architecture for both workloads isolates the training-vs-inference difference. The user is fine with a smaller model for training (8x7B fits with optimizer states; 8x22B wouldn't).

**Memory estimate (8x7B training):**
- Model: ~88GB
- Gradients: ~88GB  
- Optimizer (Adam): ~176GB
- Total: ~352GB across 8x96GB = 768GB → fits with FSDP

**Note**: `device_map='auto'` does pipeline-parallel (layer sharding), not TP/EP. For training, we'll use FSDP instead for proper gradient handling. Load model, wrap in FSDP, train.

---

## Workload B: Prefill Inference (MoE)

**Script**: New `workloads/infer_prefill.py` — Mixtral 8x22B prefill-only

- Load `mistralai/Mixtral-8x22B-v0.1` (public, ~268GB bf16)
- `device_map='auto'` distributes across 8 GPUs (~33.5GB/GPU)
- **No decode step** — forward pass only, discard outputs
- bf16, `torch.inference_mode()`
- 5 min run with same phase structure

### Batching strategy: Chunked prefill (realistic, uniform compute)

**How it works:**
1. Maintain a queue of sequences with lengths sampled from a realistic distribution:
   - Log-normal, median ~1500 tokens, range 128–8192
   - Simulates real traffic: many medium prompts, some short, some long
2. Set a **fixed token budget** per iteration (e.g., 8192 tokens total)
3. Each iteration:
   - Pull sequences from queue until we hit the token budget
   - If a sequence is too long, take a prefix chunk and save the rest for next iteration
   - Pad the batch to max sequence length in this batch
   - Forward pass through model
   - Discard KV cache (simulates shipping to decode server)
   - Cycle completed sequences out, add new ones from queue
4. Loop continuously

**Why this is the hardest case:**
- Fixed token budget → ~uniform FLOPs per iteration → ~uniform tensor core utilization
- Looks like training's uniform step structure
- No backward pass, no allreduce — but those affect NVLink/power, not tensor_sm_ratio directly

### Variant: Variable-batch prefill (easier to detect)

Optional second script `workloads/infer_prefill_variable.py`:
- Same model, but process whole sequences per batch
- Batch sizes vary: sometimes 1x 8K sequence, sometimes 16x 512 sequences
- Total FLOPs fluctuate batch-to-batch
- This creates temporal variance in tensor_sm_ratio that training doesn't have
- Easier to detect — but worth collecting to show the contrast

---

## Implementation Details

### Sequence length distribution
```python
def sample_sequence_lengths(n, min_len=128, max_len=8192, median=1500):
    """Log-normal distribution, realistic prompt lengths."""
    mu = np.log(median)
    sigma = 0.8  # spread
    lengths = np.random.lognormal(mu, sigma, n).astype(int)
    return np.clip(lengths, min_len, max_len)
```

### Chunked prefill loop (core logic)
```python
TOKEN_BUDGET = 16384
queue = deque(sample_sequence_lengths(1000))
in_flight = []  # (remaining_tokens,) for sequences being processed

while elapsed < DURATION_S:
    # Build batch: pull from queue until we hit budget
    batch_seqs = []
    tokens_used = 0
    while tokens_used < TOKEN_BUDGET and (in_flight or queue):
        if in_flight:
            seq_len = in_flight.pop(0)
        elif queue:
            seq_len = queue.popleft()
        
        chunk = min(seq_len, TOKEN_BUDGET - tokens_used)
        batch_seqs.append(chunk)
        tokens_used += chunk
        
        if seq_len > chunk:
            in_flight.append(seq_len - chunk)  # remainder for next iteration
    
    # Pad and forward
    max_len = max(batch_seqs)
    input_ids = torch.randint(0, vocab_size, (len(batch_seqs), max_len), device=device)
    attention_mask = torch.zeros_like(input_ids)
    for i, sl in enumerate(batch_seqs):
        attention_mask[i, :sl] = 1
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Refill queue if running low
    if len(queue) < 100:
        queue.extend(sample_sequence_lengths(500))
```

### Parallelism for inference
- **TP8**: All 8 GPUs active simultaneously on every layer (realistic frontier setup)
- Requires manual model sharding using `torch.distributed.tensor.parallel` or manual weight splitting
- All GPUs compute every forward pass — maximally saturates tensor cores
- NVLink traffic from activation exchanges (similar to training's TP pattern)

### Parallelism for training
- FSDP wrapping of the HF Mixtral model
- `FullyShardedDataParallel(model, sharding_strategy=ShardingStrategy.FULL_SHARD)`
- Handles gradient + optimizer sharding across 8 GPUs
- Or: simpler approach with `device_map` + gradient accumulation per-layer (less efficient but works)

---

## Files to create

| File | Description |
|------|-------------|
| `workloads/train_t20.py` | Mixtral 8x7B MoE training, FSDP, 5min |
| `workloads/infer_prefill.py` | Mixtral 8x22B chunked prefill, 5min |

## Output data files

| File | Description |
|------|-------------|
| `data/t20_telemetry.csv` | Training telemetry (DCGM 10Hz) |
| `data/prefill_telemetry.csv` | Chunked prefill telemetry |

---

## Expected outcomes / hypotheses

| Signal | Training | Chunked Prefill | Variable Prefill |
|--------|----------|-----------------|------------------|
| `tensor_sm_ratio` mean | High (~0.3-0.5) | Similar? | Similar? |
| `tensor_sm_ratio` variance | Low (uniform steps) | Low (fixed budget) | Higher (variable batch) |
| `power_w` mean | High (~350-400W) | Moderate-High? | Moderate? |
| `power_w` variance | Low | Low | Higher |
| `nvlink` pattern | Periodic (FSDP allreduce) | Minimal (pipeline fwd only) | Minimal |
| `mem_used` | Stable high (model+grad+optim) | Sawtooth? (KV alloc/free) | Sawtooth? |

**Key question**: Is `tensor_sm_ratio` indistinguishable between training and chunked prefill? If yes, we need other signals (memory pattern, NVLink, power variance).

---

## Verification

1. Run training: `torchrun --nproc_per_node=8 workloads/train_t20.py`
2. Run prefill: `python workloads/infer_prefill.py` (single process, device_map handles distribution)
3. Check CSVs exist and have ~3000+ rows each (5min at 10Hz)
4. Quick comparison: `tensor_active` and `sm_active` columns during "steady" phase
5. Plot tensor_sm_ratio time series for both — are they distinguishable?

---

## Decisions (confirmed by Atharva)

1. **Parallelism**: TP8 for inference (all GPUs active simultaneously, realistic frontier setup). Requires manual model sharding.
2. **Token budget**: 16K tokens per iteration (good saturation, moderate memory).
3. **Variants**: Chunked prefill only (hardest case first — if tensor_sm_ratio can't distinguish this, variable-batch is less interesting).

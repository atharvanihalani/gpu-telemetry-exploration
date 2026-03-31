# Workload Scripting Plan — T1 (Large Pre-Training) + I2 (Streaming Inference)

---

## T1 — Large Pre-Training

### Architecture: GPT-style Transformer

Decoder-only transformer, sized to saturate GPU memory like a real pre-training run.

**Sizing math** (DDP on 8×A100-80GB, bf16 weights+grads, fp32 AdamW states):
- Memory per param: ~12 bytes (2 weights + 2 grads + 8 Adam m+v)
- With ~20GB headroom for activations: **~5B params max per GPU**

| Config | d_model | n_layers | n_heads | Approx params | Est. GPU mem |
|---|---|---|---|---|---|
| **Medium (active)** | **3072** | **28** | **24** | **~3.2B** | **~38 GB** |
| Large (OOM) | 4096 | 32 | 32 | ~6.4B | ~77 GB |

Large config OOMed: at batch=4, seq=2048, activations alone consumed ~75GB leaving no room for AdamW m/v states. Running medium config.

- **Sequence length**: 2048 tokens
- **Batch size**: per-GPU 4–8, tuned to keep memory ~80% used
- **Optimizer**: AdamW (fp32 states)
- **Precision**: bf16
- **Gradient checkpointing**: OFF (deliberate — T5 is the checkpointing-on variant)
- **Data**: synthetic random token tensors
- **Duration**: 5 minutes steady-state; first ~30s marked as warmup and excluded from analysis

### Parallelism: DDP (PyTorch DistributedDataParallel)

Each GPU holds a full model copy. Synchronous allreduce after every backward pass = the NVLink heartbeat we're establishing as ground truth.

**Launch**: `torchrun --nproc_per_node=8 workloads/train_t1.py`

---

## I2 — Streaming Autoregressive Inference

Single-request, token-by-token generation. The "cleanest negative" — most unlike training.

- **Model**: Llama-3.1-8B (real pretrained weights from HuggingFace, ~16GB download)
- **Setup**: one model instance per GPU, 8 independent streams (no tensor parallelism)
- **Generation**: continuous autoregressive loop, long sequences (~2048 tokens), repeated for 5 minutes
- **Precision**: bf16

**Why real weights**: KV cache growth and autoregressive generation patterns are part of the telemetry signature. A randomly initialized model wouldn't exercise these faithfully.

**Note on tensor parallelism**: deferred to I3. When I3 is implemented, use tensor parallelism across all 8 GPUs — more realistic for large-model inference at this scale.

**Launch**: `python workloads/infer_i2.py`

---

## Telemetry Collection

Shared `collect_telemetry.py` module imported by both workload scripts. Runs a background thread polling pynvml every second, writes to CSV independently of the main training/inference loop.

Usage pattern in each script:
```python
collector = TelemetryCollector("data/t1_telemetry.csv")
collector.start()
# ... workload runs ...
collector.stop()
```

**Fields collected** (per GPU, per second):
- timestamp
- power_w
- sm_util_pct
- mem_used_mib / mem_total_mib
- temp_c
- pcie_tx_mb, pcie_rx_mb
- clock_sm_mhz, clock_mem_mhz
- phase label (warmup / steady / cooldown) — written by the workload script, not the collector

---

## Output Files

```
workloads/
  collect_telemetry.py   # shared background telemetry collector
  train_t1.py            # DDP pre-training (torchrun)
  infer_i2.py            # streaming inference

data/
  t1_telemetry.csv
  i2_telemetry.csv
```

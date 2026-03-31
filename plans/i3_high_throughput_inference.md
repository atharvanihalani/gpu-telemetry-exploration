# I3 — High-Throughput Batched Inference (vLLM-style)

## Goal
Continuous batching with many concurrent requests, maximizing GPU occupancy. This is the **hardest inference case to distinguish from training** — high SM util, high power, sustained load.

## Evasion angle (from the training-detection perspective)
This is not an evasion attempt — it's the worst-case legitimate inference workload. If a detector flags this as training, it has a false positive problem.

## Implementation approach
New script `infer_i3.py`. Two options:

### Option A: vLLM (most realistic)
```python
from vllm import LLM, SamplingParams

# Tensor parallelism across all 8 GPUs
llm = LLM(model="meta-llama/Llama-3.1-8B", tensor_parallel_size=8)

sampling_params = SamplingParams(temperature=0, max_tokens=512)

# Continuous request loop
while time.time() - t_start < DURATION_S:
    prompts = [generate_random_prompt() for _ in range(BATCH_SIZE)]
    outputs = llm.generate(prompts, sampling_params)
```

### Option B: Manual continuous batching (more control, no vLLM dep)
- Use tensor parallelism (split model across 8 GPUs with `accelerate` or manual sharding)
- Maintain a request queue, start new requests as old ones finish
- Use `model.generate()` with large batch sizes

### Key parameters
- Model: Llama-3.1-8B with tensor parallelism across 8 GPUs (unlike I2's 1-per-GPU)
- Concurrent requests: 32–64 (saturate GPU compute)
- `MAX_NEW_TOKENS`: 512
- Duration: 5 min

## Expected telemetry signature
- **Power**: High (~250–350W on A100, ~400–550W on H100) — approaching training levels
- **SM util**: High (~60–90%) — continuous batching keeps SMs busy
- **Memory**: Higher than I2 but lower than T1 — KV cache for many concurrent sequences, but no optimizer states (~30–40 GB)
- **NVLink**: Present but **asymmetric** — tensor parallelism generates P2P traffic for attention sharding, but it's not the symmetric allreduce pattern of DDP training
- **PCIe**: Moderate (token I/O for many concurrent requests)
- **Key question**: With power and SM util in the training range, is NVLink pattern (asymmetric TP vs symmetric allreduce) sufficient to distinguish I3 from T1?

## Hardware notes
- Tensor parallelism works on both A100 and H100 (all-to-all NVSwitch)
- vLLM supports H100 natively, including FP8 quantization (which would further change the signature)
- On H100, higher memory bandwidth means even higher throughput → even higher SM util

## Launch
```bash
# vLLM
python workloads/infer_i3.py

# Or with specific tensor parallelism
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python workloads/infer_i3.py
```

## Output
```
data/i3_telemetry.csv
```

## Dependencies
- `pip install vllm` (for Option A — large install, ~2 GB)
- Or: transformers + accelerate (for Option B)

## Complexity
Medium. vLLM handles the hard parts (continuous batching, KV cache management, tensor parallelism). Manual approach is significantly more work.

## Priority
High — this is the most important remaining inference variant. If the detector can't distinguish I3 from T1, the detector needs redesigning.

---

## Implementation notes (2026-03-31)

### What was built

`workloads/infer_i3.py` — Option A (vLLM) was chosen. Single script, no external runner needed.

### vLLM version and compatibility

- **vLLM 0.6.3.post1** installed — the latest version targeting PyTorch 2.4.x / CUDA 12.x
- Newer vLLM versions (0.6.4+) require PyTorch 2.5+; even newer (0.8+) require PyTorch 2.6+
- Installation downgraded PyTorch from 2.4.1+cu124 to 2.4.0+cu121 (bundled CUDA runtime)
  - This is benign: CUDA 12.x minor versions are forward-compatible, and the system CUDA toolkit (12.4) is still available
  - Only conflict: torchaudio 2.4.1 wants torch 2.4.1 — harmless since torchaudio is unused
- If running on a fresh node, install with: `pip install vllm==0.6.3.post1`

### Architecture decisions

- **Tensor parallelism** (`tensor_parallel_size=8`): model sharded across all 8 GPUs via vLLM's native TP. This means NVLink carries point-to-point TP communication (column/row parallel splits), not symmetric allreduce.
- **Batch size 32**: chosen to saturate GPU compute without OOM. Can be tuned up to 64 if memory allows (depends on KV cache size at max_tokens=512).
- **`ignore_eos=True`**: forces generation to always produce exactly MAX_TOKENS tokens, ensuring consistent sustained load (no early stopping on EOS).
- **`enforce_eager=False`**: allows vLLM to use CUDA graphs for higher throughput after warmup.
- **Greedy decoding** (`temperature=0`): deterministic, same as I2 for fair comparison.
- **Prompts**: pseudo-random English-like text (~64 tokens). Real-ish text avoids degenerate tokenisation that could skew memory/compute patterns.

### Telemetry integration

- `TelemetryCollector` starts *before* vLLM init (captures model loading phase on all 8 GPUs)
- Phase labels: `loading` -> `warmup` (30s) -> `steady` (remaining ~4.5 min) -> `cooldown` (5s)
- Collector runs in a daemon thread in the main process; pynvml sees all 8 GPUs regardless of vLLM's internal process structure
- Output: `data/i3_telemetry.csv` (same schema as T1/I2)

### How to run

```bash
python workloads/infer_i3.py
```

No `torchrun` needed — vLLM manages multi-GPU internally via Ray (multi-node) or multiprocessing (single-node).

### Expected runtime

- Model loading: ~30-60s (downloading if not cached, then sharding across 8 GPUs)
- Warmup: 30s
- Steady generation: ~4.5 min
- Total: ~6-7 min

### What to look for in the telemetry

Key question: can I3 be distinguished from T1 (DDP training)?

Expected distinguishing signals:
1. **NVLink pattern**: TP traffic is asymmetric point-to-point (attention sharding), not symmetric allreduce bursts. No periodic heartbeat.
2. **Memory**: lower than T1 (~30-40 GB vs ~67 GB) — no optimizer states, no gradients.
3. **Power variability**: T1 shows periodic synchronized power dips (allreduce sync). I3 should be more uniformly sustained.
4. **SM utilisation**: possibly comparable to T1 (60-90% vs 100%) — this is the signal most likely to overlap.

If power and SM util overlap between I3 and T1, the NVLink symmetry pattern becomes the critical discriminator.

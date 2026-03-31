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

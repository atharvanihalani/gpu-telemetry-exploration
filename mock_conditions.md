# Mock Workload Conditions for GPU Telemetry Differentiation

Goal: build a labeled dataset of hardware-level signatures across realistic and adversarial GPU workloads. Each condition = one "ground truth" class for the telemetry fingerprinting problem.

---

## Training Variants

### T1 — Large Pre-Training (baseline positive)
Standard data-parallel pre-training, max params that fit comfortably in 8×80GB. Allreduce every step.

**Expected signature**: sustained ~400W/GPU, SM util ~100%, NVLink simultaneous bursts every backward pass, memory near ceiling.

### T2 — Small Pre-Training
Same setup as T1 but significantly smaller model. Fits in a fraction of VRAM.

**Expected signature**: same temporal *pattern* as T1 (sustained power, NVLink heartbeat) but lower absolute levels. Tests whether pattern is a better discriminator than magnitude.

### T3 — Gradient Accumulation
Accumulate N gradient steps before running allreduce. Reduces inter-GPU communication frequency by N×.

**Evasion angle**: a detector looking for "allreduce heartbeat every ~1s" would miss this — NVLink bursts are spaced N× further apart. Power and SM util remain high throughout.

### T4 — Pipeline Parallelism
Split model layers across GPUs (GPU0 holds layers 0–k, GPU1 holds layers k–2k, etc.) instead of replicating the full model.

**Evasion angle**: NVLink traffic is a sequential wave (GPU0→1→2→...→7) rather than a simultaneous all-to-all burst. No allreduce at all — peer-to-peer activation passing only. Very different from data-parallel signature.

### T5 — Gradient Checkpointing
Discard intermediate activations during forward pass; recompute them on demand during backward. Standard memory/compute tradeoff.

**Evasion angle**: memory footprint drops significantly (could look like inference). SM utilization goes *up* (more compute per byte of memory). Could fool a memory-threshold detector.

**Implementation note**: T1 is scripted with gradient checkpointing **off** — this is deliberate. T5 is T1 with checkpointing switched on, so the two can be directly compared.

### T6 — ZeRO-3 / CPU Offloading
Optimizer states, gradients, and parameters sharded across GPUs + offloaded to CPU RAM. Only active parameters live on GPU at any time.

**Evasion angle**: nearly eliminates the "high GPU memory used" signal — GPU memory usage can look close to inference. PCIe traffic increases (CPU↔GPU transfers). The one signal that survives: NVLink is still active for parameter gathering.

---

## Inference Variants

### I1 — Standard Batched Inference (baseline negative)
Load a model, run fixed-size batches through it repeatedly.

**Expected signature**: moderate power (~150–200W), bursty SM util, low NVLink (no allreduce — tensor parallelism may generate some P2P traffic), memory steady at model weight size.

### I2 — Streaming Autoregressive (token-by-token)
Single-request, token-by-token generation. Represents the memory-bound extreme of inference.

**Expected signature**: low SM util (memory-bound, not compute-bound), power lower than batched, NVLink near zero, PCIe steady (KV cache). Looks most unlike training.

### I3 — High-Throughput Batched Inference (vLLM-style)
Many concurrent requests, continuous batching, maximizing GPU occupancy.

**Expected signature**: high SM util (approaches training levels), higher power. Closest inference mode to training superficially — but NVLink should still be sparse/asymmetric vs. allreduce heartbeat.

**Implementation note**: scripting deferred — do I2 first. When implementing I3, use tensor parallelism across all 8 GPUs (more realistic than single-model-per-GPU at this scale).

### I4 — Speculative Decoding
Small draft model generates candidate tokens; large verifier model accepts/rejects. Two models running in alternation.

**Expected signature**: two distinct power/compute "rhythms" interleaved — draft model (low power, fast) and verifier (high power, slower). Unusual temporal pattern.

---

## Adversarial / Evasion Conditions

### E1 — Power-Capped Training
Full pre-training run with `nvmlDeviceSetPowerManagementLimit` throttled to ~150W (from ~400W default).

**Evasion angle**: power signature flattens toward inference/idle range. Training slows down proportionally. Tests whether power alone is a sufficient signal, or whether NVLink + SM patterns still discriminate.

### E2 — Fake Inference Cover Traffic
Run inference on GPUs 0–3 simultaneously with training on GPUs 4–7.

**Evasion angle**: aggregate power and SM util are confused. NVLink patterns are mixed. Does a per-GPU detector still catch it? Does an aggregate-only detector get fooled?

### E3 — Intermittent Training (duty-cycled)
Train for 30s, sleep 10s, repeat.

**Evasion angle**: sustained flat power disappears — looks like bursty inference from the outside. Tests whether temporal pattern *over minutes* is required for detection, vs. per-second sampling.

### E4 — PCIe-Only Allreduce (`NCCL_P2P_DISABLE=1`)
Force NCCL to use PCIe instead of NVLink for allreduce. NVLink stays dark.

**Evasion angle**: kills the strongest training signal (NVLink heartbeat) entirely. Allreduce traffic shows up on PCIe instead. Training is slower but the NVLink-based detector sees nothing. PCIe bandwidth + SM util pattern may still be detectable.

### E5 — Smoothed Allreduce (large NCCL chunk size)
`NCCL_ALGO=RING` with a very large `NCCL_BUFFSIZE`. Stretches the allreduce communication window, turning the sharp NVLink burst into a low continuous rate.

**Evasion angle**: NVLink traffic looks less "bursty" — more like sustained background noise. Tests whether burst *shape* (sharp spike vs. flat) is load-bearing in the detector.

---

## Other Baselines

### B1 — Idle with Model Loaded
Weights fully loaded into VRAM, no computation running.

**Signature**: high memory usage, ~0% SM util, power at idle (~60–80W). Important baseline — distinguishes "has a model" from "is running a model."

### B2 — Checkpoint Save / Load
Serialize and write a large model checkpoint to disk (or load from disk).

**Signature**: PCIe and storage I/O spike, memory bandwidth spike, brief high-power moment, then drops. Short-lived anomaly; not sustained.

### B3 — Compilation / CUDA Graph Warmup
`torch.compile` trace or CUDA graph capture — one-time JIT overhead at the start of a run.

**Signature**: short SM util spike (seconds), then drops. Precedes any real workload. Worth knowing this exists so it doesn't get misclassified as a compute event.

---

## Priority Order for Experiments

| Priority | Condition | Why |
|---|---|---|
| 1 | T1 — Large Pre-Training | Clean ground-truth positive |
| 2 | I2 — Streaming Inference | Clean ground-truth negative |
| 3 | I3 — High-Throughput Inference | Hardest inference case to distinguish from training |
| 4 | E1 — Power-Capped Training | Tests whether power is necessary or redundant |
| 5 | E4 — PCIe-Only Allreduce | Tests whether NVLink is necessary |
| 6 | T3 — Gradient Accumulation | Reduces allreduce frequency — stresses temporal detector |
| 7 | E2 — Cover Traffic | Mixed-signal adversarial case |
| 8 | T4 — Pipeline Parallelism | Different inter-GPU communication topology |
| 9 | E3 — Intermittent Training | Stresses sustained-power assumption |
| 10 | Remaining | Fill in as time/budget allows |

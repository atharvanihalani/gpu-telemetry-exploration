---
name: Frontier inference landscape
description: How frontier models actually do inference — TP within NVLink domain, EP across nodes for MoE, disaggregated prefill/decode. Informs detection strategy.
type: project
---

**High-confidence constraints on frontier inference (discussed 2026-04-05):**

- **TP stays within NVLink domain** — never over IB. Latency-critical (sync every layer). On H100: 8 GPUs. On GB200 NVL72: 72 GPUs.
- **Individual transformer layers fit in 8-way TP** — even 2T MoE models have layers small enough. Wider TP only needed if NVLink domain is larger (GB200).
- **EP across nodes for large MoEs** — 256 experts (DeepSeek V3) benefit from cross-node distribution. All-to-all token shuffle over IB.
- **Continuous batching is universal** — vLLM, SGLang, TRT-LLM. Nobody does naive batching.
- **FP8 for serving** — DeepSeek V3 ships native FP8. 671B × 1 byte = 671GB fits on 2 nodes.
- **Disaggregated prefill/decode** — emerging. Prefill pool at high load could look like training (biggest false positive risk). Needs enough GPUs to split pools.

**Detection implications:**
- IB traffic during inference = EP all-to-all only (no gradient allreduce)
- IB periodicity distinguishes training (periodic DP heartbeat) from inference (irregular, demand-driven)
- IB volume: training >> inference (79 GB/s T14 vs est. 5-15 GB/s I10)
- Power: training sustained high, inference lower and more variable (decode is memory-bound)

**Why:** Understanding what inference actually looks like at frontier informs which signals reliably distinguish it from training.
**How to apply:** Focus classifier improvements on IB patterns (periodicity, volume) and power temporal profiles. NVLink alone can't distinguish TP training from TP inference.

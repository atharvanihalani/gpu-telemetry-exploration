# Workload Variants Reference — Pre-Training & Inference

Reference for building new test conditions. Covers all known variants that affect GPU telemetry signatures.

---

## Pre-Training: Three Axes of Variation

Real frontier training runs combine choices from all three axes simultaneously.

### Axis 1: Parallelism Strategy (how work is divided across GPUs)

Directly determines NVLink/network traffic signature.

| Strategy | What's split | Communication pattern | NVLink signature | Scope |
|---|---|---|---|---|
| **Data Parallelism (DP)** | Data (each GPU has full model copy) | Periodic allreduce of gradients, once per global batch | Periodic symmetric bursts — "heartbeat" | Cross-node (InfiniBand) |
| **Tensor Parallelism (TP)** | Each layer split across GPUs (column/row-wise matmul splits) | All-reduce within every layer, forward and backward | Continuous high-bandwidth hum, ~4 syncs per layer | Within one NVLink domain only |
| **Pipeline Parallelism (PP)** | Sequential layer groups across GPUs | Point-to-point activation passing (forward) and gradient passing (backward) | Staggered waves, asymmetric per-link utilization | Cross-node (tolerates higher latency) |
| **Expert Parallelism (EP)** | MoE experts distributed across GPUs | All-to-all token shuffle (data-dependent routing) | Roughly uniform but variable per-batch, 2× per MoE layer | Cross-node typically |
| **Sequence/Context Parallelism (SP/CP)** | Sequence dimension for non-matmul ops / attention | Ring-style or all-to-all during attention layers specifically | Additional traffic on top of TP, similar character | Within NVLink domain |

**Frontier configuration (e.g., 100K H100s training a 2T MoE):**
- 8-way TP within each node (mandatory — layers too wide for one GPU)
- PP across ~8-16 nodes (model too deep for one TP group)
- EP across nodes (experts distributed)
- DP across everything else (~780 replicas of the full pipeline)
- SP/CP as extension of TP

**Key insight for telemetry:** Within a single node, NVLink traffic is dominated by TP (continuous), not DP (periodic). The DP gradient allreduce goes over InfiniBand cross-node. Your daemon on one node would primarily see TP's continuous traffic pattern.

### Axis 2: Gradient Synchronization

| Method | Sync frequency | Communication reduction | Used at frontier? | Detection difficulty |
|---|---|---|---|---|
| **Synchronous SGD** | Every global batch (every step) | None (baseline) | Yes — standard today | Easy: clear periodic heartbeat |
| **Gradient accumulation** | Every N micro-batches (still every step, just larger step) | N× fewer allreduces in wall-clock terms | Yes — essentially universal | Easy: same pattern, lower frequency |
| **Local SGD / DiLoCo** | Every 100-500+ steps | 100-500× reduction | No — only tested to 10B params | Hard: rare bursts look like routine traffic |
| **Async SGD** | Never (continuous push/pull) | Eliminates synchronization | No — convergence issues at frontier | Hard: no periodic pattern at all |

**Key insight:** Everything except DiLoCo-class algorithms produces detectable periodic communication. DiLoCo is not used at frontier scale today (high confidence), but is a forward-looking threat.

**Note on gradient accumulation:** Nearly universal at frontier scale. The global batch (e.g., 4M tokens) is split into micro-batches (e.g., 32K tokens). Each GPU processes micro-batches sequentially: forward → backward → accumulate gradients locally. Allreduce happens only after all micro-batches are processed. Mathematically equivalent to one giant batch.

### Axis 3: Memory Management

| Strategy | Tradeoff | Effect on telemetry | Used at frontier? |
|---|---|---|---|
| **ZeRO Stage 1** | Shard optimizer states → less memory, minimal extra communication | Nearly invisible | Yes — very common |
| **ZeRO Stage 2** | Shard optimizer + gradients → less memory, reduce-scatter instead of allreduce | Minimal change | Yes — common |
| **ZeRO Stage 3 / FSDP** | Shard everything → much less memory, but per-layer all-gathers required | *More* NVLink traffic | Yes, but less common when TP is already used |
| **Gradient checkpointing** | Discard activations, recompute in backward → less memory, ~33% more compute | Higher power, same communication | Yes — nearly universal |
| **CPU/NVMe offloading** | Move optimizer states to CPU RAM → much less GPU memory, heavy PCIe traffic | Lower GPU memory, high PCIe, 2-4× slowdown | Only when model won't fit otherwise |
| **Mixed precision (bf16/fp16 + fp32 master)** | Lower memory for forward/backward, fp32 for optimizer | Universal constant, not a variant | Yes — universal |

**Key insight for verification:** Axis 3 strategies mostly make training *easier* to detect (more communication, more compute). CPU offloading can hide the memory signal but at severe performance cost, and doesn't affect power or NVLink signatures.

---

## Inference Variants

### Batching Strategies

| Variant | Description | Prevalence | Telemetry signature |
|---|---|---|---|
| **Naive batching** | Collect N requests, process together, wait for longest | Nobody at scale | Bursty, idle gaps between batches |
| **Continuous batching** | Slot new requests into finished slots without waiting | Universal (vLLM, TGI, TRT-LLM) | Smoother utilization, less idle time, demand-driven |
| **Chunked prefill** | Interleave prompt processing with token generation for other requests | Increasingly standard | More uniform compute, less variance |

### Memory Optimization

| Variant | Description | Prevalence | Telemetry signature |
|---|---|---|---|
| **KV cache** | Store key/value tensors from previous tokens to avoid recomputation | Universal, non-optional | Memory grows linearly with sequence length × batch size |
| **PagedAttention** | Manage KV cache like virtual memory pages, eliminate fragmentation | Very widely adopted (vLLM) | Higher memory efficiency → more concurrent requests → higher sustained utilization |
| **KV cache quantization** | Store KV cache in int8/fp8 instead of bf16 | Increasingly common | Less memory per request, more concurrent requests |
| **Prefix caching** | Cache KV states for common prompt prefixes (system prompts, few-shot) | Widespread | Reduced compute for cached prefixes, more variable workload |

### Parallelism for Inference

| Variant | Description | Prevalence | Telemetry signature |
|---|---|---|---|
| **Tensor parallelism** | Split each layer across GPUs, same as training TP | Standard for large models | Continuous NVLink traffic — **known false positive risk** (I3) |
| **Pipeline parallelism** | Split layers sequentially across GPUs/nodes | Less common than in training | Sequential point-to-point traffic |
| **Expert parallelism (MoE)** | Distribute experts across GPUs for MoE models | Required for large MoE inference | Token shuffle traffic, lower throughput than training |

### Advanced Inference Techniques

| Variant | Description | Prevalence | Telemetry signature |
|---|---|---|---|
| **Speculative decoding** | Small draft model generates candidates, large verifier checks in parallel | Used in production (Google confirmed for Gemini) | Two interleaved compute rhythms (draft=low power, verifier=high power) |
| **Disaggregated prefill/decode** | Separate prefill (compute-bound) and decode (memory-bound) onto different GPU pools | Emerging, medium adoption | Different GPUs look very different: some bursty high compute, others sustained low power |
| **Multi-node long-context inference** | Split KV cache across nodes for very long contexts (100K+), ring attention | Niche, only for very long contexts | NVLink/InfiniBand traffic during attention, could resemble training TP |
| **Distilled/quantized serving** | Serve a smaller distilled or quantized version of the trained model | Very common | Lower power, memory, and NVLink than expected for lab's "frontier" model |
| **Model routing** | Route easy queries to smaller models, hard queries to frontier model | Common | Different GPUs running different models, variable utilization |

---

## Classifier Implications Summary

### Signals that are robust across all configurations:
- **Power (sustained high)** — fundamental to computation. TP training, DP training, hybrid — all produce sustained high power. Cannot be hidden without stopping computation.
- **Tensor/SM ratio** — backward pass is always matmul-dominated regardless of parallelism strategy.

### Signals that need rethinking:
- **NVLink autocorrelation** — designed for DP's periodic heartbeat. At frontier scale, intra-node NVLink is dominated by TP (continuous, not periodic). May need to shift from "periodic structure" to "sustained high bandwidth" detection, possibly as compound rule (high NVLink AND high power) to distinguish from TP inference.

### Key discriminator that always holds:
**Inference has no backward pass, no gradients, no optimizer step.** This means:
- No periodic DP gradient allreduce (the heartbeat)
- Lower total NVLink volume (forward-only vs forward+backward)
- Power driven by user demand (variable/spiky over hours) vs training (flat for weeks)
- Duration: inference is continuous/elastic, training is fixed multi-week allocations

### Priority test conditions to add:
1. **TP training** (8-way Megatron-style) — compare NVLink profile to I3 (TP inference)
2. **TP+DP training** — hybrid, see how DP allreduce looks on top of TP background
3. **MoE inference with EP** — expert routing traffic during inference
4. **High-load continuous batching with chunked prefill** — worst-case inference for false positives

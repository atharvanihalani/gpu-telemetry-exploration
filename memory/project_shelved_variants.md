---
name: Shelved workload variants
description: Multi-node variants discussed but deprioritized — revisit when we have more compute or need to stress-test specific detection limits
type: project
---

**Large gradient accumulation multi-node (T16?)** — shelved 2026-04-04. 5-line diff on T10 (wrap fwd/bwd in a loop). Tests whether autocorrelation detection breaks when allreduce period exceeds the 0.2-5s analysis window. Cheap to collect but more of a stress test than a realistic scenario. Revisit if autocorrelation window tuning becomes a priority.

**TP=4 / exotic sub-8 TP configs** — shelved 2026-04-04. Unusual for H100 (full 8-GPU NVLink domain makes TP=8 standard). More relevant for GB200 NVL72 or smaller models. Revisit when we have different hardware.

**Hybrid FSDP+DP (HSDP)** — shelved 2026-04-04. FSDP within node (8 GPUs, NVLink) + DP across nodes (IB). More realistic than full FSDP at frontier, but IB signature is just periodic DP allreduce again (same as T10). Quick variant on top of full FSDP (T15) — revisit after T15 data is in.

**PP+DP across nodes** — shelved 2026-04-04. 2-stage PP × 2-way DP needs 4+ nodes to be realistic. On 2 nodes the pipeline is too coarse (50% bubble).

**DiLoCo / Local SGD** — shelved 2026-04-04. Not used at frontier scale, unverified beyond 10B params. Revisit if frontier adoption changes.

**MoE expert count / routing variants** — discussed 2026-04-04. Different expert counts (64 top-1, 4 top-2), different dense-to-MoE ratios. Quantitative variations on T12/T14, not fundamentally different signatures. Lower priority.

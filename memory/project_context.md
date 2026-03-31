---
name: Project context and current state
description: What the GPU telemetry project is building toward and where it currently stands
type: project
---

Goal: build a technical foundation for verifying that AI labs have paused pre-training runs during a coordinated slowdown. Core question: can you distinguish training from inference (or idle) from GPU telemetry signals?

**Current state (as of 2026-03-31):**
- Session 1: passive exploration — pynvml/DCGM baseline, NVLink topology, idle metrics
- Session 2: active fingerprinting — ran T1 (DDP training) and I2 (streaming inference) mock workloads, collected labeled 5-min telemetry CSVs, built comparison notebook

**Key finding from session 2**: zero signal overlap between T1 and I2 at 1Hz. The allreduce heartbeat (periodic synchronized power dips across all 8 GPUs) is clearly visible in training and completely absent in inference.

**Next priority**: evasion conditions from mock_conditions.md:
1. E1 — power-capped training
2. E4 — PCIe-only allreduce (NCCL_P2P_DISABLE=1)
3. T3 — gradient accumulation
4. I3 — high-throughput batched inference

**Why:** sessions 1-2 established clean poles. Evasion conditions stress-test whether any single signal is sufficient, or whether a combination is needed.

**How to apply:** frame new work around the evasion/detection framing — what signal survives each evasion, what combination is hardest to simultaneously fake.

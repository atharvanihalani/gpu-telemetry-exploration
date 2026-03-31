---
name: Project context and current state
description: What the GPU telemetry project is building toward and where it currently stands
type: project
---

Goal: build a technical foundation for verifying that AI labs have paused pre-training runs during a coordinated slowdown. Core question: can you distinguish training from inference (or idle) from GPU telemetry signals?

**Current state (as of 2026-03-31):**
- Session 1: passive exploration — pynvml/DCGM baseline, NVLink topology, idle metrics
- Session 2: active fingerprinting — ran T1 (DDP training) and I2 (streaming inference), collected labeled 5-min telemetry CSVs, built comparison notebook
- Session 3: implemented all remaining workload scripts (T2–T6, I3, I4, E1–E5, B1). Moved to H100 node. Made code hardware-agnostic. None of the new workloads have been run yet — scripts are ready, telemetry collection is pending.

**Key finding from session 2**: zero signal overlap between T1 and I2 at 1Hz. The allreduce heartbeat (periodic synchronized power dips across all 8 GPUs) is clearly visible in training and completely absent in inference.

**Next priority**: run the implemented workloads and collect telemetry, in this order:
1. E1 — power-capped training (tests most-cited detection signal)
2. E4 — PCIe-only allreduce (tests strongest detection signal — NVLink heartbeat)
3. I3 — high-throughput inference (hardest inference case, closest to training)
4. T3 — gradient accumulation (stresses temporal pattern detector)
5. E3 — intermittent training (breaks sustained-power assumption)
6. Then build a multi-condition comparison analysis notebook

**Why:** sessions 1–2 established clean poles. Session 3 built the tooling. Session 4+ should collect data and answer: what signal survives each evasion, what combination is hardest to simultaneously fake.

**How to apply:** frame new work around collecting telemetry and building comparison analysis. The scripts are all written — focus shifts to running experiments and analyzing results.

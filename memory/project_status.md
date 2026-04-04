---
name: Project status after session 10
description: T13-T15 collected (TP+PP, TP+EP+DP, full FSDP). PP training hardest to detect. Plans in plans/ folder.
type: project
---

**Completed (sessions 1-10, as of 2026-04-04):**
- 14 single-node conditions (H100, DCGM 10Hz): T1-T6, E2-E5, I2-I4, B1
- 6 multi-node conditions (2x H100, DCGM+IB+BMC): T10-T15
- Three independent telemetry collectors: DCGM (10Hz), IB sysfs (10Hz), BMC/IPMI (2s)

**Session 10 new conditions and key results (node 0 data, steady-state):**

| Condition | Power | tensor_sm_ratio | NVLink tx | IB tx | Classifier |
|---|---|---|---|---|---|
| T13 (TP+PP) | 418W | 0.235 | 50 GB/s | 0.52 GB/s | tensor_ratio MISS (0.235 < 0.25), power barely catches (418 > 400) |
| T14 (TP+EP+DP) | 480W | 0.369 | 32 GB/s | 79 GB/s | Easy catch on power + tensor_ratio |
| T15 (Full FSDP) | 417W | 0.657 | 23 GB/s | 25.6 GB/s | tensor_ratio catches (0.657), power barely (417 > 400) |

**Session 10 key findings:**
- PP training (T13) is hardest to detect — pipeline bubble dilutes power and tensor_ratio below/near thresholds
- T15 FSDP has highest tensor_sm_ratio (0.657) of any workload — FSDP keeps tensor cores proportionally very busy
- T14 has highest IB traffic (79 GB/s) — DP allreduce of MoE expert gradients
- T15 is first condition with continuous non-periodic IB traffic (per-layer all-gather/reduce-scatter)
- `destroy_process_group()` hangs with multi-node FSDP too, not just composable TP

**Implementation notes:**
- T13 needed `foreach=False` in AdamW (mixed DTensor/Tensor without replicate)
- T13 agent wrote `os.exit(0)` instead of `os._exit(0)` — caught and fixed
- Plans saved in `plans/` folder (t13_tp_pp.md, t14_tp_moe_dp.md)
- Node 1 data needs to be copied from claude-two for T13, T14, T15

**Why:** Each parallelism strategy produces a distinct signature. PP is the most adversarial for the current classifier.
**How to apply:** Classifier needs lower power threshold or compound rules to catch PP training. IB patterns (P2P vs allreduce vs continuous) are a new detection axis.

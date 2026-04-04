---
name: Project status after session 9
description: Multi-node telemetry on Hyperbolic 2xH100 cluster — T10/T11/T12 collected, IB+BMC collectors added, nvlink_autocorr fails on TP
type: project
---

**Completed (sessions 1-9, as of 2026-04-04):**
- 14 single-node conditions (H100, DCGM 10Hz): T1-T6, E2-E5, I2-I4, B1
- 3 multi-node conditions (2x H100, DCGM+IB+BMC): T10 (DDP), T11 (TP+DP), T12 (MoE EP+DP)
- Three independent telemetry collectors: DCGM (10Hz), IB sysfs (10Hz), BMC/IPMI (2s)
- Consistency checks validated: BMC SYS_POWER matches DCGM per-GPU power, temps agree

**Key session 9 finding:**
Different parallelism strategies produce dramatically different NVLink signatures:
- T10 (pure DP): periodic heartbeat (18 GB/s) — classifier catches easily
- T11 (TP+DP): continuous high bandwidth (56 GB/s) — `nvlink_autocorr` rule FAILS (no periodicity)
- T12 (MoE EP+DP): data-dependent variable traffic from all-to-all (TBD)

**Classifier status (3 rules, needs rethinking):**
- `mean_power > 400W` — still robust across all parallelism strategies
- `tensor_sm_ratio > 0.25` — works for DP, borderline for TP (0.15 in T11)
- `nvlink_autocorr_peak > 0.3` — fails for TP training (continuous, not periodic)
- Need new rules: sustained high NVLink bandwidth, IB traffic patterns

**Infrastructure:** Hyperbolic bare-metal, 2x H100 SXM5, 8x NDR 400G IB, BMC accessible, BlueField-2 present

**Why:** Multi-node with realistic parallelism (TP+DP, MoE) breaks single-node detection assumptions.
**How to apply:** Classifier needs rethinking for frontier-realistic configurations. IB collector fills the gap for cross-node detection.

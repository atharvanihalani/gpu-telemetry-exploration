---
name: Project status after session 8
description: Current state of GPU telemetry exploration — 14 conditions collected, 3-rule classifier (12/14), I3 false positive, E4 miss
type: project
---

**Completed (sessions 1-8, as of 2026-04-03):**
- Baseline telemetry collection (A100 + H100)
- 14 workload conditions collected at DCGM 10Hz on H100: T1-T6, E2, E3, E4, E5, I2, I3, I4, B1
- Multi-condition comparison analysis (12 plots in `plots/`, redone with all 14 conditions in session 8)
- Threshold-based training classifier with 3 rules in `classifier/`

**Classifier status (3 rules, OR logic, 12/14 correct):**
- `mean_power > 400W` — sustained high compute
- `tensor_sm_ratio > 0.25` — matmul-dominated workload
- `nvlink_autocorr_peak > 0.3` — periodic NVLink heartbeat
- **Removed in session 8:** `power_std_temporal > 40W` — unprincipled, won't generalize
- **Known false positive**: I3 (vLLM tensor-parallel inference) triggers nvlink_autocorr due to per-layer activation exchanges at steady rate
- **Known miss**: E4 (PCIe-only allreduce) dodges all 3 rules, but imposes ~10x throughput penalty — unrealistic evasion

**Shelved workloads:**
- E1: Power capping deprioritized — unrealistic evasion (kills throughput), blocked on RunPod (needs CAP_SYS_ADMIN)

**Next steps:**
- Rethink classifier: compound rules for I3 false positive, possibly PCIe traffic rule for E4
- Threshold sensitivity sweep
- Write up findings for verification proposal

**Why:** Open-ended exploration to understand what signals distinguish training from inference and how robust detection can be under adversarial conditions.
**How to apply:** The evasion conditions (E2-E5) are the real test. Each breaks one signal but not all — multi-signal detection remains robust, though rule thresholds need refinement.

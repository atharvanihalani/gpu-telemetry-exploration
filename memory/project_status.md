---
name: Project status after session 7
description: Current state of GPU telemetry exploration — 13 conditions collected, classifier has 4 rules with 1 known false positive
type: project
---

**Completed (sessions 1-7, as of 2026-04-02):**
- Baseline telemetry collection (A100 + H100)
- 13 workload conditions collected at DCGM 10Hz on H100: T1-T6, E2, E3, E4, E5, I2, I3, B1
- Multi-condition comparison analysis (12 plots in `plots/`, needs redo with E2/E5)
- Threshold-based training classifier with 4 rules in `classifier/`

**Classifier status (4 rules, OR logic, 12/13 correct):**
- `mean_power > 400W` — sustained high compute
- `tensor_sm_ratio > 0.25` — matmul-dominated workload
- `power_std_temporal > 40W` — step-cycle oscillation
- `nvlink_autocorr_peak > 0.3` — periodic NVLink heartbeat (NEW, session 7)
- **Known false positive**: I3 (vLLM tensor-parallel inference) triggers nvlink_autocorr due to per-layer activation exchanges at steady rate
- **Near miss**: E2 aggregate tensor_sm_ratio is 0.242 (threshold 0.25) — cover traffic nearly dodges it

**Shelved workloads (need fixes):**
- E1: RunPod blocks power limit API — needs SM clock cap approach instead
- I4: transformers meta tensor bug with 2 models per GPU

**Next steps:**
- Redo comparison notebook with all 13 conditions
- Rethink classifier rules — consider compound rules (e.g., autocorrelation AND power) to fix I3 false positive
- Run threshold sensitivity sweep
- Write up findings for verification proposal

**Why:** Open-ended exploration to understand what signals distinguish training from inference and how robust detection can be under adversarial conditions.
**How to apply:** The evasion conditions (E1-E5) are the real test. Each breaks one signal but not all — multi-signal detection remains robust, though rule thresholds need refinement.

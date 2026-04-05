---
name: Project status after session 11
description: Session 11 (2026-04-05) — multi-node inference planning, I10 torchrun script ready, DeepSeek V3 downloaded, SGLang multi-node broken
type: project
---

**Completed (sessions 1-11, as of 2026-04-05):**
- 14 single-node conditions (H100, DCGM 10Hz): T1-T6, E2-E5, I2-I4, B1
- 6 multi-node training conditions (2x H100, DCGM+IB+BMC): T10-T15
- I10 inference script ready (torchrun-based MoE TP+EP, debug run passed)
- DeepSeek V3 downloaded on both nodes (642 GB each)

**Session 11 key work:**
- Analyzed frontier inference landscape: TP within NVLink domain only, EP across nodes for MoE, disaggregated prefill/decode emerging
- Attempted SGLang multi-node (Gloo bug, 3 failed attempts) and vLLM (torch version conflict)
- Pivoted to torchrun-based `infer_i10.py`: T14 architecture minus backward pass, cross-node EP (16 experts across 16 GPUs)
- Set up SSH between nodes (can launch both from single session)
- Synced repo and venv to node 1

**Next steps:**
- Run I10 with telemetry enabled (toy MoE model)
- Figure out DeepSeek V3 loading strategy for real inference telemetry
- Power capping experiment (`nvidia-smi -pl`)
- Gradient compression (PowerSGD)

**Why:** Multi-node inference is the key gap — need to compare inference vs training IB/NVLink signatures for MoE models.
**How to apply:** Use torchrun for all multi-node workloads. Avoid SGLang/vLLM multi-node until Gloo bug is fixed.

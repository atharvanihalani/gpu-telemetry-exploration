---
name: Hardware and environment setup
description: RunPod node details, DCGM setup, and environment requirements for telemetry collection
type: reference
---

**Nodes:** RunPod 8-GPU pods, either A100 SXM4-80GB or H100 SXM5-80GB. ~$12/hr. Sessions 1-2 on A100, sessions 3-4 on H100.

**DCGM setup (required before any workload):**
```bash
nv-hostengine
dcgmi discovery -l
```

**Environment requirements:**
- `HF_TOKEN` env var needed for inference workloads (Llama-3.1-8B is gated)
- vLLM 0.18.1 (pinned for PyTorch compat)
- DCGM Python bindings at `/usr/local/dcgm/bindings/python3/`

**Data to save before stopping pod:** All CSVs in `data/`, plots in `plots/`, notebooks.

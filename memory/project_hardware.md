---
name: Hardware and environment setup
description: Hyperbolic H100 cluster details, node setup requirements, and environment for telemetry collection
type: reference
---

**Current cluster (session 9+):** 2x H100 SXM5 nodes on Hyperbolic, 8 GPUs each, 16 total. 8x NDR 400G InfiniBand inter-node. Node IPs may change between allocations — verify in CLAUDE.md.

**Previous cluster (sessions 1-8):** RunPod A100/H100 single-node pods.

**Fresh node setup (Ubuntu 24.04):**
```bash
sudo apt-get update -q && sudo apt-get install -y datacenter-gpu-manager python3.12-venv ipmitool
sudo nv-hostengine
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install nvidia-ml-py3 pandas matplotlib seaborn transformers accelerate huggingface_hub
```

**Why:** At the start of a new conversation on a fresh node, torch/venv may not be installed. Check before running workloads. Run `python3 -c "import torch"` as a quick test.

**Other requirements:**
- `HF_TOKEN` env var needed for inference workloads (Llama-3.1-8B is gated)
- `ipmitool` needed for BMC telemetry collection
- `nv-hostengine` must be running for DCGM
- DCGM Python bindings at `/usr/local/dcgm/bindings/python3/`

**Data to save before losing nodes:** All CSVs in `data/`, plots in `plots/`, notebooks.

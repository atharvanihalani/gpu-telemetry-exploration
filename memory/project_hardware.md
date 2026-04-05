---
name: Hardware and environment setup
description: Hyperbolic H100 cluster details, SSH between nodes, node setup, torch version requirements
type: reference
---

**Current cluster (session 9+):** 2x H100 SXM5 nodes on Hyperbolic, 8 GPUs each, 16 total. 8x NDR 400G InfiniBand inter-node, 8 IB interfaces per node (ib0-ib7). Node IPs are ephemeral — check `ip addr` on each node.

**SSH between nodes:** Configured in `~/.ssh/config`. From node 0: `ssh node1`, `ssh node1-ib`. Reverse works. Enables launching both nodes from a single Claude session.

**Previous cluster (sessions 1-8):** RunPod A100/H100 single-node pods.

**Fresh node setup (Ubuntu 24.04):**
```bash
sudo apt-get update -q && sudo apt-get install -y datacenter-gpu-manager python3.12-venv ipmitool
sudo nv-hostengine
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install torch torchvision  # default PyPI, NOT cu124 index
pip install nvidia-ml-py3 pandas matplotlib seaborn transformers accelerate huggingface_hub
```

**Critical: torch version matching.** Both nodes MUST have the same torch version for torchrun multi-node. Use default PyPI (`pip install torch`), not `--index-url cu124`. As of session 11: both at 2.9.1+cu128.

**SGLang/vLLM warning:** Don't install both in the same venv. They fight over torch/triton versions. SGLang 0.5.9 multi-node has a Gloo bug — use torchrun instead.

**Syncing repo to node 1:**
```bash
rsync -az --exclude='.git' --exclude='data/' --exclude='.claude/' ~/gpu-telemetry-exploration/ node1:~/gpu-telemetry-exploration/
```

**Other requirements:**
- `HF_TOKEN` in `.env` file — needed for gated models (Llama-3.1-8B)
- `nv-hostengine` must be running for DCGM
- DeepSeek V3 (642 GB) downloaded on both nodes at `~/.cache/huggingface/hub/deepseek-v3/`

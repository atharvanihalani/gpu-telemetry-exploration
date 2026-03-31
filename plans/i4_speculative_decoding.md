# I4 — Speculative Decoding

## Goal
Run speculative decoding: a small draft model generates candidate tokens, a large verifier model accepts/rejects them in parallel. Tests whether the two-model interleaved pattern creates a distinctive telemetry fingerprint.

## Implementation approach
New script `infer_i4.py`. Use HuggingFace's `assisted_generation` or manual speculative decoding loop.

### Option A: HuggingFace assisted generation
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Draft model (small, fast)
draft_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B", torch_dtype=torch.bfloat16, device_map={"": device}
)

# Verifier model (large)
verifier_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B", torch_dtype=torch.bfloat16, device_map={"": device}
)

# Speculative decoding
outputs = verifier_model.generate(
    input_ids,
    assistant_model=draft_model,
    max_new_tokens=512,
    do_sample=False,
)
```

### Setup options
- **Single GPU per pair**: Draft + verifier share one GPU (simplest, 4 active GPU pairs)
- **Dedicated GPUs**: Draft on GPUs 0–3, verifier on GPUs 4–7 (more complex, tests asymmetric load)

### Key parameters
- Draft: Llama-3.2-1B (~2 GB)
- Verifier: Llama-3.1-8B (~16 GB)
- Both fit on a single 80GB GPU (~18 GB total)
- `num_assistant_tokens`: 5–10 (draft generates this many before verification)
- Duration: 5 min

## Expected telemetry signature
- **Power**: Two interleaved levels — low (draft generating) and high (verifier checking). Creates a distinctive "staircase" or "pulse" pattern.
- **SM util**: Alternating — low during draft (small model, memory-bound), high during verification (larger model, batch forward pass)
- **Memory**: ~18 GB per GPU (both models loaded)
- **NVLink**: Near zero (no inter-GPU communication, each GPU independent)
- **Key question**: Is the alternating power/SM pattern distinguishable from other bursty workloads?

## Hardware notes
- Both models fit easily on A100 or H100
- Llama-3.2-1B is gated on HuggingFace — may need `huggingface-cli login`
- Alternative draft model if gating is an issue: use a random small transformer

## Launch
```bash
python workloads/infer_i4.py
```

## Output
```
data/i4_telemetry.csv
```

## Dependencies
- transformers >= 4.35 (for `assistant_model` support)
- Llama-3.2-1B access (gated) or substitute small model

## Complexity
Low-medium. HuggingFace handles the speculative decoding loop. Main work is loading two models per GPU and managing the generation loop.

---

## Implementation Notes (added during implementation)

### Draft model choice

**Primary**: `meta-llama/Llama-3.2-1B` — same tokenizer as Llama-3.1-8B (both Llama 3 family), so standard assisted generation works directly. ~2 GB in bf16. This is the ideal pairing: HuggingFace's own Llama recipes demonstrate Llama-3.2-1B/3B as the draft model for Llama-3.1-8B/70B with 1.5-2x speedups. **However, this model is gated** — requires a HuggingFace token and Meta license approval.

**Fallback**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` — ungated, ~1.1B params, but uses Llama 2's tokenizer (different from Llama 3.1). To handle the tokenizer mismatch, the script uses **Universal Assisted Decoding (UAD)**, available in transformers >= 4.46.0. UAD translates tokens between tokenizer vocabularies via text re-encoding and longest-common-subsequence alignment. Acceptance rate will be lower than the same-tokenizer pairing, but the telemetry pattern (alternating draft/verify compute bursts) will still be present — and that's what matters for this project.

The script auto-detects accessibility via `huggingface_hub.model_info()` and falls back gracefully.

### Setup chosen: single GPU per pair

Each GPU loads both the draft model (~2 GB) and verifier model (~16 GB) in bf16, totaling ~18 GB — fits comfortably in 80 GB with room for KV caches and activations. 8 independent streams, one per GPU, same threading model as I2. No tensor parallelism, no inter-GPU communication.

### API details

- Uses `model.generate(assistant_model=draft_model, ...)` — the standard HuggingFace assisted generation API (available since transformers 4.35)
- UAD mode additionally passes `tokenizer=verifier_tokenizer, assistant_tokenizer=draft_tokenizer` (transformers >= 4.46)
- Greedy decoding (`do_sample=False`) for deterministic output
- Dynamic speculation is the default since transformers 4.45 — `num_assistant_tokens` is automatically adjusted based on acceptance rate (heuristic schedule)
- 64-token random prompt, 512 max new tokens per request, continuous loop for 5 minutes

### Environment

- PyTorch: 2.4.1+cu124
- transformers: needs >= 4.35 (basic assisted generation), >= 4.46 (UAD fallback)
- Both models loaded in bf16 to match I2's setup

### File

`workloads/infer_i4.py` — launch with `python workloads/infer_i4.py`

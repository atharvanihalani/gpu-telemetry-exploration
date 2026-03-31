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

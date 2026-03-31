"""
I4 — Speculative Decoding Inference

Loads a small draft model (~1B params) and a large verifier model
(Llama-3.1-8B) on each GPU and runs speculative decoding via
HuggingFace's assisted generation API. Both models share a single GPU
(~18 GB total), leaving plenty of headroom on 80 GB A100/H100.

The expected telemetry signature is an alternating power/compute pattern:
low during draft generation (small model, memory-bound) and higher during
verification (large model forward pass). This creates a distinctive
"pulse" or "staircase" in the SM utilization and power traces that
differs from both vanilla autoregressive inference (I2) and training (T1).

8 independent GPU streams (one draft+verifier pair per GPU, no tensor
parallelism) — same threading model as I2.

Draft model selection:
  - Primary:  meta-llama/Llama-3.2-1B  (same tokenizer as Llama-3.1-8B,
              ideal acceptance rate — but gated, requires HF token + Meta
              license approval)
  - Fallback: TinyLlama/TinyLlama-1.1B-Chat-v1.0  (ungated, different
              tokenizer — uses Universal Assisted Decoding / UAD via the
              tokenizer + assistant_tokenizer params, requires
              transformers >= 4.46)

The script auto-detects which draft model is accessible and falls back
gracefully.

Usage:
    python workloads/infer_i4.py

Config (edit at top of file):
    VERIFIER_MODEL_ID   HuggingFace model ID for verifier
    DRAFT_MODEL_ID      HuggingFace model ID for draft (primary)
    FALLBACK_DRAFT_ID   HuggingFace model ID for draft (fallback)
    DURATION_S          total wall-clock run time
    WARMUP_S            initial phase excluded from "steady" analysis
    MAX_NEW_TOKENS      tokens to generate per request before resetting
    PROMPT_LEN          input prompt length (tokens)
"""

import os
import sys
import time
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workloads.collect_telemetry import TelemetryCollector

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VERIFIER_MODEL_ID  = "meta-llama/Llama-3.1-8B"
DRAFT_MODEL_ID     = "meta-llama/Llama-3.2-1B"        # gated — same tokenizer
FALLBACK_DRAFT_ID  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # ungated — different tokenizer

DURATION_S         = 5 * 60
WARMUP_S           = 30
MAX_NEW_TOKENS     = 512     # tokens per generation request before resetting
PROMPT_LEN         = 64      # synthetic prompt length in tokens

OUTPUT_CSV         = "data/i4_telemetry.csv"


# ---------------------------------------------------------------------------
# Draft model resolution
# ---------------------------------------------------------------------------

def resolve_draft_model() -> tuple:
    """Determine which draft model to use and whether UAD is needed.

    Returns (draft_model_id, use_uad) where use_uad=True means the draft
    model has a different tokenizer and we need to pass tokenizer +
    assistant_tokenizer to generate().
    """
    # Try the primary (same-tokenizer) draft first
    try:
        from huggingface_hub import model_info
        info = model_info(DRAFT_MODEL_ID)
        # If we get here without error, the model metadata is accessible.
        # For gated models, we can read metadata but may not download weights
        # without a token. We optimistically assume that if the metadata is
        # readable, we have access (the download will fail later if not).
        print(f"[draft] Primary draft model accessible: {DRAFT_MODEL_ID}")
        return DRAFT_MODEL_ID, False
    except Exception as e:
        print(f"[draft] Primary draft model not accessible ({e})")
        print(f"[draft] Falling back to: {FALLBACK_DRAFT_ID} (UAD mode)")
        return FALLBACK_DRAFT_ID, True


# ---------------------------------------------------------------------------
# Per-GPU worker
# ---------------------------------------------------------------------------

def run_gpu(gpu_id: int, duration_s: float,
            verifier_model_id: str, draft_model_id: str,
            verifier_tokenizer, draft_tokenizer,
            use_uad: bool,
            prompt_len: int, max_new_tokens: int,
            ready_event: threading.Event):
    """Load both models on a single GPU and run speculative decoding."""
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    # --- Load draft model (small, fast) ---
    try:
        print(f"[GPU {gpu_id}] loading draft model: {draft_model_id} ...")
        draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_id,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
        )
        draft_model.eval()
        print(f"[GPU {gpu_id}] draft model loaded")
    except Exception as e:
        print(f"[GPU {gpu_id}] DRAFT LOAD ERROR: {e}")
        ready_event.set()
        return

    # --- Load verifier model (large) ---
    try:
        print(f"[GPU {gpu_id}] loading verifier model: {verifier_model_id} ...")
        verifier_model = AutoModelForCausalLM.from_pretrained(
            verifier_model_id,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
        )
        verifier_model.eval()
        print(f"[GPU {gpu_id}] verifier model loaded")
    except Exception as e:
        print(f"[GPU {gpu_id}] VERIFIER LOAD ERROR: {e}")
        ready_event.set()
        return

    ready_event.set()

    vocab_size = verifier_model.config.vocab_size
    pad_token_id = verifier_tokenizer.eos_token_id if verifier_tokenizer.eos_token_id else 0

    t_start = time.time()
    step = 0
    total_tokens = 0

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        while time.time() - t_start < duration_s:
            # Generate random prompt tokens (using verifier vocab range)
            input_ids = torch.randint(
                100, vocab_size - 100,
                (1, prompt_len),
                device=device,
            )
            attention_mask = torch.ones_like(input_ids)

            # Build generate() kwargs
            gen_kwargs = dict(
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                assistant_model=draft_model,
                pad_token_id=pad_token_id,
            )

            # Universal Assisted Decoding: pass both tokenizers when
            # draft and verifier use different tokenizers
            if use_uad:
                gen_kwargs["tokenizer"] = verifier_tokenizer
                gen_kwargs["assistant_tokenizer"] = draft_tokenizer

            out = verifier_model.generate(input_ids, **gen_kwargs)

            tokens_generated = out.shape[1] - prompt_len
            total_tokens += tokens_generated
            step += 1

            if gpu_id == 0 and step % 5 == 0:
                elapsed = time.time() - t_start
                print(f"[GPU 0] step={step}  tokens={total_tokens}  "
                      f"tok/s={total_tokens / elapsed:.1f}  elapsed={elapsed:.0f}s")

    print(f"[GPU {gpu_id}] done — {step} requests, {total_tokens} tokens generated")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    n_gpus = torch.cuda.device_count()
    draft_model_id, use_uad = resolve_draft_model()

    print(f"Found {n_gpus} GPUs")
    print(f"Verifier: {VERIFIER_MODEL_ID}")
    print(f"Draft:    {draft_model_id}  (UAD={'yes' if use_uad else 'no'})")
    print(f"Duration: {DURATION_S}s  |  Prompt: {PROMPT_LEN} tokens  "
          f"|  Max new: {MAX_NEW_TOKENS} tokens")

    # Load tokenizers once in main thread
    print("Loading verifier tokenizer ...")
    verifier_tokenizer = AutoTokenizer.from_pretrained(VERIFIER_MODEL_ID)

    if use_uad:
        print("Loading draft tokenizer (UAD mode) ...")
        draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_id)
    else:
        # Same tokenizer — no need for separate draft tokenizer
        draft_tokenizer = None

    # Start telemetry collector
    collector = TelemetryCollector(OUTPUT_CSV)
    collector.start()
    collector.set_phase("loading")

    # Launch one thread per GPU
    ready_events = [threading.Event() for _ in range(n_gpus)]
    threads = []
    for gpu_id in range(n_gpus):
        t = threading.Thread(
            target=run_gpu,
            args=(gpu_id, DURATION_S, VERIFIER_MODEL_ID, draft_model_id,
                  verifier_tokenizer, draft_tokenizer, use_uad,
                  PROMPT_LEN, MAX_NEW_TOKENS, ready_events[gpu_id]),
            daemon=True,
        )
        threads.append(t)
        t.start()

    # Wait for all GPUs to finish loading both models
    print("Waiting for all GPUs to load draft + verifier models ...")
    for ev in ready_events:
        ev.wait()
    print("All GPUs ready — starting warmup phase")
    collector.set_phase("warmup")

    # Switch to steady after warmup
    time.sleep(WARMUP_S)
    collector.set_phase("steady")

    # Wait for all workers to finish
    for t in threads:
        t.join()

    collector.set_phase("cooldown")
    time.sleep(5)
    collector.stop()
    print("Done.")


if __name__ == "__main__":
    main()

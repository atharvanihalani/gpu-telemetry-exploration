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

8 independent GPU processes (one draft+verifier pair per GPU, no tensor
parallelism). Uses multiprocessing instead of threading to avoid GIL
contention in HF's assisted generation Python-side coordination.

Draft model: meta-llama/Llama-3.2-1B (same tokenizer as Llama-3.1-8B,
gated — requires HF token + Meta license approval).

Previous attempts with cross-tokenizer drafts (Qwen3-0.6B, Qwen2.5-0.5B)
hit RoPE dimension mismatches in transformers' UAD path. Using a same-family
draft avoids UAD entirely.

Usage:
    python workloads/infer_i4.py

Config (edit at top of file):
    VERIFIER_MODEL_ID   HuggingFace model ID for verifier
    DRAFT_MODEL_ID      HuggingFace model ID for draft
    DURATION_S          total wall-clock run time
    WARMUP_S            initial phase excluded from "steady" analysis
    MAX_NEW_TOKENS      tokens to generate per request before resetting
    PROMPT_LEN          input prompt length (tokens)
"""

import os
import sys
import time
import multiprocessing as mp

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workloads.collect_telemetry import TelemetryCollector

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VERIFIER_MODEL_ID  = "meta-llama/Llama-3.1-8B"
DRAFT_MODEL_ID     = "meta-llama/Llama-3.2-1B"

DURATION_S         = 5 * 60
WARMUP_S           = 30
MAX_NEW_TOKENS     = 512
PROMPT_LEN         = 64

OUTPUT_CSV         = "data/i4_telemetry.csv"


# ---------------------------------------------------------------------------
# Per-GPU worker (runs in its own process)
# ---------------------------------------------------------------------------

def run_gpu(gpu_id: int, duration_s: float,
            verifier_model_id: str, draft_model_id: str,
            prompt_len: int, max_new_tokens: int,
            ready_barrier: mp.Barrier):
    """Load both models on a single GPU and run speculative decoding."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # --- Load draft model (small, fast) ---
    try:
        print(f"[GPU {gpu_id}] loading draft model: {draft_model_id} ...")
        draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_id,
            dtype=torch.bfloat16,
            device_map={"": device},
        )
        draft_model.eval()
        print(f"[GPU {gpu_id}] draft model loaded")
    except Exception as e:
        print(f"[GPU {gpu_id}] DRAFT LOAD ERROR: {e}")
        ready_barrier.abort()
        return

    # --- Load verifier model (large) ---
    try:
        print(f"[GPU {gpu_id}] loading verifier model: {verifier_model_id} ...")
        verifier_model = AutoModelForCausalLM.from_pretrained(
            verifier_model_id,
            dtype=torch.bfloat16,
            device_map={"": device},
        )
        verifier_model.eval()
        print(f"[GPU {gpu_id}] verifier model loaded")
    except Exception as e:
        print(f"[GPU {gpu_id}] VERIFIER LOAD ERROR: {e}")
        ready_barrier.abort()
        return

    tokenizer = AutoTokenizer.from_pretrained(verifier_model_id)
    vocab_size = verifier_model.config.vocab_size
    pad_token_id = tokenizer.eos_token_id or 0

    # Wait for all GPUs to finish loading
    try:
        ready_barrier.wait()
    except mp.BrokenBarrierError:
        print(f"[GPU {gpu_id}] barrier broken — another GPU failed to load")
        return

    t_start = time.time()
    step = 0
    total_tokens = 0

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        while time.time() - t_start < duration_s:
            input_ids = torch.randint(
                100, vocab_size - 100,
                (1, prompt_len),
                device=device,
            )
            attention_mask = torch.ones_like(input_ids)

            out = verifier_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                assistant_model=draft_model,
                pad_token_id=pad_token_id,
            )

            tokens_generated = out.shape[1] - prompt_len
            total_tokens += tokens_generated
            step += 1

            if gpu_id == 0 and step % 5 == 0:
                elapsed = time.time() - t_start
                print(f"[GPU 0] step={step}  tokens={total_tokens}  "
                      f"tok/s={total_tokens / elapsed:.1f}  elapsed={elapsed:.0f}s")

    print(f"[GPU {gpu_id}] done — {step} requests, {total_tokens} tokens generated")


# ---------------------------------------------------------------------------
# Main (parent process — runs telemetry collector)
# ---------------------------------------------------------------------------

def main():
    mp.set_start_method("spawn", force=True)

    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPUs")
    print(f"Verifier: {VERIFIER_MODEL_ID}")
    print(f"Draft:    {DRAFT_MODEL_ID}")
    print(f"Duration: {DURATION_S}s  |  Prompt: {PROMPT_LEN} tokens  "
          f"|  Max new: {MAX_NEW_TOKENS} tokens")

    # Start telemetry collector (parent process, sees all GPUs via DCGM)
    collector = TelemetryCollector(OUTPUT_CSV)
    collector.start()
    collector.set_phase("loading")

    # Barrier: n_gpus workers + 1 parent (to coordinate phase switch)
    ready_barrier = mp.Barrier(n_gpus + 1)

    # Launch one process per GPU
    procs = []
    for gpu_id in range(n_gpus):
        p = mp.Process(
            target=run_gpu,
            args=(gpu_id, DURATION_S, VERIFIER_MODEL_ID, DRAFT_MODEL_ID,
                  PROMPT_LEN, MAX_NEW_TOKENS, ready_barrier),
        )
        procs.append(p)
        p.start()

    # Wait for all GPUs to finish loading
    print("Waiting for all GPUs to load draft + verifier models ...")
    try:
        ready_barrier.wait()
    except mp.BrokenBarrierError:
        print("ERROR: one or more GPUs failed to load models — aborting")
        for p in procs:
            p.terminate()
        collector.stop()
        sys.exit(1)

    print("All GPUs ready — starting warmup phase")
    collector.set_phase("warmup")

    time.sleep(WARMUP_S)
    collector.set_phase("steady")

    # Wait for all workers to finish
    for p in procs:
        p.join()

    collector.set_phase("cooldown")
    time.sleep(5)
    collector.stop()
    print("Done.")


if __name__ == "__main__":
    main()

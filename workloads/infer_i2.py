"""
I2 — Streaming Autoregressive Inference

Loads Llama-3.1-8B and runs token-by-token generation continuously for
DURATION_S seconds across all 8 GPUs (one model instance per GPU,
independent streams — no tensor parallelism).

Establishes ground-truth inference telemetry signature: memory-bound,
bursty SM util, near-zero NVLink, power well below training levels.

Usage:
    python workloads/infer_i2.py

Config (edit at top of file):
    MODEL_ID     HuggingFace model ID
    DURATION_S   total wall-clock run time
    MAX_NEW_TOKENS  tokens to generate per request before resetting
    PROMPT_LEN   input prompt length (tokens)
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
MODEL_ID        = "meta-llama/Llama-3.1-8B"
DURATION_S      = 5 * 60
WARMUP_S        = 30
MAX_NEW_TOKENS  = 512     # tokens per generation request before resetting
PROMPT_LEN      = 64      # synthetic prompt length in tokens

OUTPUT_CSV      = "data/i2_telemetry.csv"


# ---------------------------------------------------------------------------
# Per-GPU worker
# ---------------------------------------------------------------------------

def run_gpu(gpu_id: int, duration_s: float, model_id: str,
            tokenizer, prompt_len: int, max_new_tokens: int,
            ready_event: threading.Event):
    """Load model on a single GPU and generate continuously."""
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    try:
        print(f"[GPU {gpu_id}] loading {model_id} ...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
        )
        model.eval()
        print(f"[GPU {gpu_id}] model loaded")
    except Exception as e:
        print(f"[GPU {gpu_id}] LOAD ERROR: {e}")
        ready_event.set()  # unblock main thread even on failure
        return

    ready_event.set()

    vocab_size = model.config.vocab_size
    pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else 0
    # Build a dummy attention mask to suppress the pad==eos warning
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

            out = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
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
# Main
# ---------------------------------------------------------------------------

def main():
    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPUs")
    print(f"Model: {MODEL_ID}")
    print(f"Duration: {DURATION_S}s  |  Prompt: {PROMPT_LEN} tokens  "
          f"|  Max new: {MAX_NEW_TOKENS} tokens")

    # Load tokenizer once in main thread (avoids import races in workers)
    print(f"Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Start telemetry collector
    # When TELEMETRY_DISABLED is set, skip collector (orchestrator owns it).
    collector = None
    if not os.environ.get("TELEMETRY_DISABLED"):
        collector = TelemetryCollector(OUTPUT_CSV)
        collector.start()
        collector.set_phase("loading")

    # Launch one thread per GPU
    ready_events = [threading.Event() for _ in range(n_gpus)]
    threads = []
    for gpu_id in range(n_gpus):
        t = threading.Thread(
            target=run_gpu,
            args=(gpu_id, DURATION_S, MODEL_ID, tokenizer,
                  PROMPT_LEN, MAX_NEW_TOKENS, ready_events[gpu_id]),
            daemon=True,
        )
        threads.append(t)
        t.start()

    # Wait for all GPUs to finish loading
    print("Waiting for all GPUs to load model ...")
    for ev in ready_events:
        ev.wait()
    print("All GPUs ready — starting warmup phase")
    if collector:
        collector.set_phase("warmup")

    # Switch to steady after warmup
    time.sleep(WARMUP_S)
    if collector:
        collector.set_phase("steady")

    # Wait for all workers to finish
    for t in threads:
        t.join()

    if collector:
        collector.set_phase("cooldown")
        time.sleep(5)
        collector.stop()
    print("Done.")


if __name__ == "__main__":
    main()

"""
B1 — Idle with Model Loaded

Loads Llama-3.1-8B onto all 8 GPUs, then sits idle for DURATION_S seconds.
Establishes baseline for "has a model loaded but not running" — high memory,
zero compute. Important to avoid false positives on idle inference servers.

Usage:
    python workloads/baseline_b1.py
"""

import os
import sys
import time
import threading

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workloads.collect_telemetry import TelemetryCollector

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID    = "meta-llama/Llama-3.1-8B"
DURATION_S  = 5 * 60
WARMUP_S    = 30
OUTPUT_CSV  = "data/b1_telemetry.csv"


# ---------------------------------------------------------------------------
# Per-GPU loader
# ---------------------------------------------------------------------------

def load_model(gpu_id: int, model_id: str, ready_event: threading.Event):
    """Load model onto a single GPU."""
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

    ready_event.set()

    # Keep reference alive so model stays in GPU memory
    # Block until main thread signals exit
    while not getattr(ready_event, "_exit", False):
        time.sleep(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPUs")
    print(f"Model: {MODEL_ID}")
    print(f"Duration: {DURATION_S}s (idle after loading)")

    # Start telemetry collector
    collector = TelemetryCollector(OUTPUT_CSV)
    collector.start()
    collector.set_phase("loading")

    # Launch one thread per GPU to load the model
    ready_events = [threading.Event() for _ in range(n_gpus)]
    threads = []
    for gpu_id in range(n_gpus):
        t = threading.Thread(
            target=load_model,
            args=(gpu_id, MODEL_ID, ready_events[gpu_id]),
            daemon=True,
        )
        threads.append(t)
        t.start()

    # Wait for all GPUs to finish loading
    print("Waiting for all GPUs to load model ...")
    for ev in ready_events:
        ev.wait()
    print("All GPUs ready — entering idle phase")

    collector.set_phase("warmup")
    time.sleep(WARMUP_S)

    collector.set_phase("steady")
    print(f"Idling for {DURATION_S - WARMUP_S}s ...")
    time.sleep(DURATION_S - WARMUP_S)

    collector.set_phase("cooldown")
    time.sleep(5)
    collector.stop()

    # Signal threads to exit
    for ev in ready_events:
        ev._exit = True

    print("Done.")


if __name__ == "__main__":
    main()

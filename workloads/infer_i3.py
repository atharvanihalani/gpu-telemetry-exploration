"""
I3 — High-Throughput Batched Inference (vLLM)

Uses vLLM with tensor parallelism across all 8 GPUs to run continuous
high-throughput inference.  This is the hardest inference case to
distinguish from training: continuous batching keeps SM utilisation high,
power draw sustained, and tensor-parallelism generates NVLink traffic.

Unlike I2 (one model per GPU, independent streams), I3 shards a single
model across all GPUs via tensor parallelism and feeds a continuous
stream of large batches to maximise GPU occupancy.

Key differences from T1 (training):
  - NVLink traffic is asymmetric (TP point-to-point) not symmetric allreduce
  - No optimizer states in memory (weights + KV cache only)
  - No periodic gradient-sync heartbeat in power trace
  - Memory footprint lower (~30-40 GB vs ~67 GB for T1)

Usage:
    python workloads/infer_i3.py

Config (edit constants at top of file):
    MODEL_ID          HuggingFace model ID
    DURATION_S        total wall-clock run time (generation loop)
    WARMUP_S          seconds of generation to label as warmup
    BATCH_SIZE        number of prompts per generate() call
    MAX_TOKENS        max new tokens per request
    PROMPT_LEN        input prompt length (tokens)
    TP_SIZE           tensor parallel size (number of GPUs)
    GPU_MEM_UTIL      fraction of GPU memory vLLM may use
"""

import os
import sys
import time
import random
import string

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID        = "meta-llama/Llama-3.1-8B"
DURATION_S      = 5 * 60        # 5 minutes of generation
WARMUP_S        = 30            # first 30s labelled warmup
COOLDOWN_S      = 5             # telemetry continues after generation stops
BATCH_SIZE      = 32            # prompts per generate() call
MAX_TOKENS      = 512           # new tokens per request
PROMPT_LEN      = 64            # approximate prompt length in tokens (~words)
TP_SIZE         = 8             # tensor parallel across all GPUs
GPU_MEM_UTIL    = 0.90          # vLLM gpu_memory_utilization

OUTPUT_CSV      = "data/i3_telemetry.csv"


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

# Pool of topic prefixes for variety — content doesn't matter for telemetry,
# but real-ish text avoids degenerate tokenisation edge cases.
_TOPIC_PREFIXES = [
    "Explain the history of",
    "Write a detailed analysis of",
    "Describe the process of",
    "Compare and contrast",
    "Summarize the key findings about",
    "Discuss the implications of",
    "Provide a comprehensive overview of",
    "Analyze the relationship between",
    "Evaluate the effectiveness of",
    "Outline the main arguments for and against",
]

_TOPIC_SUBJECTS = [
    "quantum computing and its applications in cryptography",
    "the development of renewable energy sources worldwide",
    "machine learning algorithms for natural language processing",
    "the economic impact of climate change on coastal cities",
    "advances in materials science for semiconductor fabrication",
    "distributed systems and consensus protocols",
    "the evolution of programming languages over the past fifty years",
    "neural network architectures for computer vision tasks",
    "supply chain optimization using operations research methods",
    "the role of international cooperation in space exploration",
    "protein folding prediction and drug discovery pipelines",
    "autonomous vehicle safety standards and regulatory frameworks",
    "large-scale data center cooling and energy efficiency",
    "the mathematical foundations of modern cryptographic systems",
    "ocean current modelling and weather prediction accuracy",
    "the philosophical implications of artificial general intelligence",
]


def random_prompt() -> str:
    """Generate a pseudo-random prompt of roughly PROMPT_LEN tokens.

    Combines a topic prefix + subject + padding words to reach the
    target length.  The exact token count varies but is close enough
    for telemetry purposes (vLLM handles variable-length inputs natively).
    """
    prefix = random.choice(_TOPIC_PREFIXES)
    subject = random.choice(_TOPIC_SUBJECTS)
    base = f"{prefix} {subject}."

    # Rough estimate: 1 word ~ 1.3 tokens.  Pad with filler words if needed.
    target_words = int(PROMPT_LEN / 1.3)
    base_words = len(base.split())
    if base_words < target_words:
        padding_words = target_words - base_words
        filler = " ".join(
            "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))
            for _ in range(padding_words)
        )
        base = f"{base} Additional context: {filler}."

    return base


def generate_batch(batch_size: int) -> list[str]:
    """Generate a batch of random prompts."""
    return [random_prompt() for _ in range(batch_size)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- Telemetry collector (must start before vLLM touches GPUs) --------
    # Insert project root so we can import the shared collector.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from workloads.collect_telemetry import TelemetryCollector

    collector = TelemetryCollector(OUTPUT_CSV)
    collector.start()
    collector.set_phase("loading")

    # ---- vLLM engine init -------------------------------------------------
    from vllm import LLM, SamplingParams

    print(f"Initialising vLLM engine ...")
    print(f"  Model           : {MODEL_ID}")
    print(f"  Tensor parallel : {TP_SIZE}")
    print(f"  GPU mem util    : {GPU_MEM_UTIL}")
    print(f"  Batch size      : {BATCH_SIZE}")
    print(f"  Max new tokens  : {MAX_TOKENS}")
    print(f"  Duration        : {DURATION_S}s  (warmup {WARMUP_S}s)")

    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=TP_SIZE,
        gpu_memory_utilization=GPU_MEM_UTIL,
        dtype="auto",               # bf16 on A100/H100
        trust_remote_code=False,
        enforce_eager=False,         # allow CUDA graphs
    )
    print("vLLM engine ready.")

    sampling_params = SamplingParams(
        temperature=0,          # greedy (deterministic) — same as I2
        max_tokens=MAX_TOKENS,
        ignore_eos=True,        # always generate exactly MAX_TOKENS tokens
    )

    # ---- Warmup phase -----------------------------------------------------
    collector.set_phase("warmup")
    print(f"Starting warmup ({WARMUP_S}s) ...")

    t_gen_start = time.time()
    batch_idx = 0
    total_requests = 0
    total_tokens_generated = 0

    # Run generation continuously; phase switches to "steady" after WARMUP_S
    phase_switched = False

    while True:
        elapsed = time.time() - t_gen_start
        if elapsed >= DURATION_S:
            break

        # Switch from warmup to steady
        if not phase_switched and elapsed >= WARMUP_S:
            collector.set_phase("steady")
            phase_switched = True
            print("Switched to steady-state phase.")

        # Generate a batch
        prompts = generate_batch(BATCH_SIZE)
        t_batch_start = time.time()

        outputs = llm.generate(prompts, sampling_params)

        t_batch_end = time.time()
        batch_dur = t_batch_end - t_batch_start

        # Count tokens generated
        batch_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        total_requests += len(outputs)
        total_tokens_generated += batch_tokens
        batch_idx += 1

        # Progress report every 5 batches
        if batch_idx % 5 == 0:
            elapsed = time.time() - t_gen_start
            tps = total_tokens_generated / elapsed if elapsed > 0 else 0
            print(
                f"[batch {batch_idx}]  requests={total_requests}  "
                f"tokens={total_tokens_generated}  "
                f"tok/s={tps:.0f}  "
                f"batch_dur={batch_dur:.1f}s  "
                f"elapsed={elapsed:.0f}s / {DURATION_S}s"
            )

    # ---- Cooldown ---------------------------------------------------------
    elapsed_total = time.time() - t_gen_start
    print(f"\nGeneration complete: {batch_idx} batches, "
          f"{total_requests} requests, "
          f"{total_tokens_generated} tokens in {elapsed_total:.0f}s")
    print(f"Average throughput: "
          f"{total_tokens_generated / elapsed_total:.0f} tok/s")

    collector.set_phase("cooldown")
    time.sleep(COOLDOWN_S)
    collector.stop()
    print(f"Telemetry saved to {OUTPUT_CSV}")
    print("Done.")


if __name__ == "__main__":
    main()

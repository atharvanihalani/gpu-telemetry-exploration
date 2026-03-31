"""
E2 — Fake Inference Cover Traffic

Orchestrates simultaneous inference (GPUs 0–3) and training (GPUs 4–7) on the
same node. Tests whether splitting workloads across GPU subsets can confuse
aggregate-level telemetry detectors.

The orchestrator owns the single telemetry collector (sees all 8 GPUs).
Both sub-processes run with TELEMETRY_DISABLED=1 so they don't create
their own collectors.

Usage:
    python workloads/run_e2.py

Output:
    data/e2_telemetry.csv
"""

import os
import sys
import subprocess
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workloads.collect_telemetry import TelemetryCollector

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DURATION_S      = 5 * 60   # total orchestrator run time
WARMUP_S        = 30       # warmup phase after processes initialize
INIT_WAIT_S     = 10       # time to let sub-processes initialize
COOLDOWN_S      = 5        # cooldown after sub-processes finish/killed

TRAIN_GPUS      = "4,5,6,7"
INFER_GPUS      = "0,1,2,3"
TRAIN_NPROC     = 4
MASTER_PORT     = "29501"  # avoid conflict with default 29500

OUTPUT_CSV      = "data/e2_telemetry.csv"


def kill_proc(proc, name):
    """Terminate a subprocess, escalating to SIGKILL if needed."""
    if proc.poll() is not None:
        return  # already dead
    print(f"[e2] sending SIGTERM to {name} (pid={proc.pid})")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        print(f"[e2] {name} did not exit, sending SIGKILL")
        proc.kill()
        proc.wait()


def main():
    print("=" * 60)
    print("E2 — Fake Inference Cover Traffic")
    print(f"  Training GPUs: {TRAIN_GPUS}  ({TRAIN_NPROC}-GPU DDP)")
    print(f"  Inference GPUs: {INFER_GPUS}")
    print(f"  Duration: {DURATION_S}s  |  Warmup: {WARMUP_S}s")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Start the single telemetry collector (sees all 8 GPUs)
    # ------------------------------------------------------------------
    collector = TelemetryCollector(OUTPUT_CSV)
    collector.start()
    collector.set_phase("loading")

    # ------------------------------------------------------------------
    # 2. Launch training sub-process on GPUs 4–7
    # ------------------------------------------------------------------
    train_env = os.environ.copy()
    train_env["CUDA_VISIBLE_DEVICES"] = TRAIN_GPUS
    train_env["TELEMETRY_DISABLED"] = "1"

    train_cmd = [
        "torchrun",
        f"--nproc_per_node={TRAIN_NPROC}",
        f"--master_port={MASTER_PORT}",
        "workloads/train_t1.py",
    ]
    print(f"[e2] launching training: {' '.join(train_cmd)}")
    print(f"[e2]   CUDA_VISIBLE_DEVICES={TRAIN_GPUS}")
    train_proc = subprocess.Popen(train_cmd, env=train_env)

    # ------------------------------------------------------------------
    # 3. Launch inference sub-process on GPUs 0–3
    # ------------------------------------------------------------------
    infer_env = os.environ.copy()
    infer_env["CUDA_VISIBLE_DEVICES"] = INFER_GPUS
    infer_env["TELEMETRY_DISABLED"] = "1"

    infer_cmd = ["python", "workloads/infer_i2.py"]
    print(f"[e2] launching inference: {' '.join(infer_cmd)}")
    print(f"[e2]   CUDA_VISIBLE_DEVICES={INFER_GPUS}")
    infer_proc = subprocess.Popen(infer_cmd, env=infer_env)

    # ------------------------------------------------------------------
    # 4. Phase management
    # ------------------------------------------------------------------
    t_start = time.time()

    # Give sub-processes time to initialize (model loading, DDP init, etc.)
    print(f"[e2] waiting {INIT_WAIT_S}s for sub-processes to initialize ...")
    time.sleep(INIT_WAIT_S)

    # Check neither died during init
    if train_proc.poll() is not None:
        print(f"[e2] ERROR: training process died during init (rc={train_proc.returncode})")
        kill_proc(infer_proc, "inference")
        collector.set_phase("error")
        collector.stop()
        sys.exit(1)
    if infer_proc.poll() is not None:
        print(f"[e2] ERROR: inference process died during init (rc={infer_proc.returncode})")
        kill_proc(train_proc, "training")
        collector.set_phase("error")
        collector.stop()
        sys.exit(1)

    # Warmup phase
    collector.set_phase("warmup")
    print(f"[e2] warmup for {WARMUP_S}s ...")
    warmup_end = t_start + INIT_WAIT_S + WARMUP_S
    while time.time() < warmup_end:
        # Check if either process died
        if train_proc.poll() is not None:
            print(f"[e2] training process exited early (rc={train_proc.returncode})")
            kill_proc(infer_proc, "inference")
            break
        if infer_proc.poll() is not None:
            print(f"[e2] inference process exited early (rc={infer_proc.returncode})")
            kill_proc(train_proc, "training")
            break
        time.sleep(1)

    # Steady phase
    both_alive = train_proc.poll() is None and infer_proc.poll() is None
    if both_alive:
        collector.set_phase("steady")
        print("[e2] steady phase — collecting telemetry ...")

        deadline = t_start + DURATION_S
        while time.time() < deadline:
            if train_proc.poll() is not None:
                print(f"[e2] training process exited (rc={train_proc.returncode})")
                kill_proc(infer_proc, "inference")
                break
            if infer_proc.poll() is not None:
                print(f"[e2] inference process exited (rc={infer_proc.returncode})")
                kill_proc(train_proc, "training")
                break
            time.sleep(1)

    # ------------------------------------------------------------------
    # 5. Cleanup: kill any remaining sub-processes, cooldown, stop
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    print(f"[e2] elapsed={elapsed:.0f}s — cleaning up")

    kill_proc(train_proc, "training")
    kill_proc(infer_proc, "inference")

    collector.set_phase("cooldown")
    time.sleep(COOLDOWN_S)
    collector.stop()

    train_rc = train_proc.returncode
    infer_rc = infer_proc.returncode
    print(f"[e2] training rc={train_rc}, inference rc={infer_rc}")
    print(f"[e2] telemetry saved to {OUTPUT_CSV}")
    print("[e2] done.")


if __name__ == "__main__":
    main()

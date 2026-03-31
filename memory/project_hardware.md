---
name: hardware_switching
description: Atharva switches between A100 and H100 RunPod nodes — scripts are hardware-agnostic, but absolute telemetry values differ
type: project
---

Atharva flips between A100 SXM4-80GB and H100 SXM5-80GB RunPod nodes (both 8-GPU, 80GB, all-to-all NVSwitch).

**Why:** Node availability and cost vary on RunPod; both GPU types serve the project's needs.

**How to apply:**
- Never hardcode A100-specific values (watts, NVLink link count, clock speeds) in code or analysis.
- Telemetry CSVs now have a `gpu_model` column to distinguish which hardware produced each row.
- Compare patterns (heartbeat, synchronization) rather than absolute values across GPU types.
- E1 power cap must be TDP-relative (~20–25% of TDP), not a fixed wattage.
- See `hardware_notes.md` in the repo for the full A100 vs H100 comparison and switching checklist.

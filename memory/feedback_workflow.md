---
name: Preferred workflow for new workload variants
description: Chat → agent plans (background) → review → agent implements (worktree) → review before running. Overlap aggressively.
type: feedback
---

When building new workload variants (or any multi-step implementation task), use this pipeline:

1. **Chat** — discuss the variant with Atharva (what, why, expected signature, detection value)
2. **Agent plans** — kick off a planning agent in the background while continuing to chat about the next thing
3. **Quick review** — skim the plan, then surface any design decisions or open questions where Atharva's input would be valuable (hardware constraints, realism of config, expected signatures, etc.). Get alignment before implementation.
4. **Agent implements** — in a worktree, from the plan. Can overlap with planning the next variant.
5. **Review diff** — careful review before merging. Watch for subtle bugs (deadlocks, dtype mismatches in comms, gradient flow issues) that produce garbage telemetry rather than crashing.

Always save the plan to `plans/` (e.g., `plans/t13_tp_pp.md`) so there's a persistent record alongside the code.

For training workload implementations: merge directly after review without waiting for explicit user approval. These are self-contained scripts that don't affect existing code.

**Why:** Atharva wants to crank out many variants per session. Overlapping planning/implementation for different variants maximizes throughput. The chat→plan→implement→review layering keeps Atharva in the loop on direction while delegating mechanical work.

**How to apply:** Default to this structure when Atharva asks to build new workloads. Don't wait for one variant to finish before starting the next. Use worktree isolation so parallel agents don't conflict.

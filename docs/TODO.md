# Open TODOs

## Classifier improvements
- [ ] Rethink NVLink rule: `nvlink_autocorr` fails on TP training (continuous, not periodic). Need sustained-bandwidth rule or compound rule.
- [ ] `tensor_sm_ratio` threshold borderline for TP (0.15 in T11 vs 0.25 threshold)
- [ ] Compound rules for I3 false positive (nvlink_autocorr AND power threshold)
- [ ] PCIe traffic rule for E4 miss
- [ ] IB traffic rules for multi-node detection
- [ ] Threshold sensitivity sweep

## Multi-node (session 9)
- [x] T10 — 16-GPU DDP across 2 nodes (collected)
- [x] T11 — TP+DP hybrid, 8-way TP within node (collected)
- [x] IB telemetry collector (`collect_ib.py`) — 10Hz sysfs counters
- [x] BMC telemetry collector (`collect_bmc.py`) — 2s IPMI sensors
- [x] Consistency checks validated (BMC power/temps match DCGM)
- [ ] T12 — MoE EP+DP (in progress)
- [x] T13 — TP+PP, 8-way TP within node, 2-stage PP across nodes (collected)
- [x] T14 — TP+EP+DP, frontier MoE config: TP on attn/dense FFN, EP on MoE, DP across nodes (collected)
- [x] T15 — Full FSDP across 16 GPUs, continuous IB traffic (collected)
- [ ] Multi-node inference (I10+ — TP inference across nodes, MoE inference with EP)
- [ ] Multi-node evasion conditions (TBD)
- [ ] EP across nodes (all-to-all over IB instead of NVLink) — needs more nodes/experts to be realistic
- [ ] Hybrid FSDP+DP (HSDP) — FSDP within node, DP across. More realistic but IB is periodic (shelved)
- [ ] Large gradient accumulation multi-node — stretches allreduce period, tests autocorrelation window (shelved)
- [ ] Exotic sub-8 TP configs (TP=4, PP=2, DP=2) — unusual for H100, revisit on different hardware (shelved)
- [ ] DiLoCo / Local SGD — not frontier-scale today, revisit if adoption changes (shelved)
- [ ] PP+DP across nodes (2-stage PP × 2-way DP) — needs 4+ nodes to be realistic (shelved)
- [ ] Merge node 1 data (CSVs need to be copied from node 2)

## Analysis
- [ ] Multi-node comparison notebook (T10 vs T11 vs T12 NVLink/IB patterns)
- [ ] Consistency check plots (DCGM power vs BMC SYS_POWER, DCGM temp vs BMC GPU_PROC)

## Write-up
- [ ] Findings summary for verification proposal

## Known issues
- `dist.destroy_process_group()` hangs with composable TP+DP (`replicate` + `parallelize_module`). Workaround: `os._exit(0)` after flushing collectors. Affects T11+.
- `dist.destroy_process_group()` also hangs with multi-node FSDP (T15). Same workaround needed — add `os._exit(0)` to T15.
- T13 `os.exit(0)` typo was fixed to `os._exit(0)` (agent wrote wrong one initially).

## Minor / cleanup
- [ ] T2 missing `TELEMETRY_DISABLED` env var check (T1 has it, matters if T2 is used in an orchestrator like E2)
- [x] ~~`classify.ipynb` sweep section references `Thresholds().power_std` — removed in session 8~~ (resolved)

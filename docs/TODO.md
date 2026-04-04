# Open TODOs

## Classifier improvements
- [ ] Compound rules for I3 false positive (nvlink_autocorr AND power threshold)
- [ ] PCIe traffic rule for E4 miss
- [ ] Threshold sensitivity sweep

## Multi-node (session 9)
- [x] T10 script — 16-GPU DDP across 2 nodes
- [x] IB telemetry collector (`collect_ib.py`) — 10Hz sysfs counters
- [x] BMC telemetry collector (`collect_bmc.py`) — 2s IPMI sensors
- [ ] Run T10 and collect data
- [ ] Multi-node pipeline parallelism (T11?)
- [ ] Multi-node FSDP (T12?)
- [ ] Multi-node evasion conditions (TBD)
- [ ] Consistency check analysis: DCGM power vs BMC SYS_POWER, DCGM temp vs BMC GPU_PROC

## Write-up
- [ ] Findings summary for verification proposal

## Known issues
- `dist.destroy_process_group()` hangs with composable TP+DP (`replicate` + `parallelize_module`). Workaround: `os._exit(0)` after flushing collectors. Affects T11+. Root cause: straggling NCCL collectives during cleanup.

## Minor / cleanup
- [ ] T2 missing `TELEMETRY_DISABLED` env var check (T1 has it, matters if T2 is used in an orchestrator like E2)
- [ ] `classify.ipynb` sweep section references `Thresholds().power_std` — removed in session 8, will throw `AttributeError`

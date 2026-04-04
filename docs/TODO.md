# Open TODOs

## Classifier improvements
- [ ] Compound rules for I3 false positive (nvlink_autocorr AND power threshold)
- [ ] PCIe traffic rule for E4 miss
- [ ] Threshold sensitivity sweep

## Multi-node (session 9)
- [ ] 16-GPU DDP (T1-16) — allreduce over NVLink + InfiniBand
- [ ] Multi-node pipeline parallelism
- [ ] Multi-node FSDP
- [ ] Multi-node evasion conditions (TBD)

## Write-up
- [ ] Findings summary for verification proposal

## Minor / cleanup
- [ ] T2 missing `TELEMETRY_DISABLED` env var check (T1 has it, matters if T2 is used in an orchestrator like E2)

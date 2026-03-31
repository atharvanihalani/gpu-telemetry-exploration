# T5 — Gradient Checkpointing

## Goal
Run T1 with gradient checkpointing enabled. Discards intermediate activations during forward pass and recomputes them during backward. Trades memory for compute — tests whether a memory-threshold detector can be fooled.

## Evasion angle
Memory footprint drops significantly because intermediate activations are not stored. GPU memory usage could approach inference levels for the same model. SM utilization goes *up* (more compute per step due to recomputation). A detector that uses "high memory = training" would be fooled.

## Implementation approach
Fork `train_t1.py` → `train_t5.py`. Add `torch.utils.checkpoint.checkpoint` to each transformer block. Optionally try a larger model (since memory freed up by checkpointing might allow the 6.4B config that OOMed in T1).

### Key code change
```python
from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    # ... same as T1 ...
    pass

class GPT(nn.Module):
    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False)
        x = self.ln_f(x)
        return self.head(x)
```

### Model configs to try
| Config | d_model | n_layers | Params | T1 mem (no ckpt) | T5 mem (ckpt) |
|---|---|---|---|---|---|
| Medium (same as T1) | 3072 | 28 | ~3.2B | ~38 GB | ~20 GB |
| Large (OOMed in T1) | 4096 | 32 | ~6.4B | ~77 GB (OOM) | ~35–40 GB |

The large config is the interesting one — it OOMed without checkpointing but should fit with it. This would produce a 6.4B training run with ~35–40GB memory, which is well below T1's 66GB.

### Key parameters
- Model config: try both medium (3.2B) and large (6.4B)
- Everything else same as T1 (DDP, same batch size, same duration)

## Expected telemetry signature
- **Power**: Similar to T1 or slightly higher (recomputation adds ~33% more FLOPs)
- **SM util**: ~100% (same or higher than T1)
- **Memory**: Significantly lower — ~20 GB for 3.2B, ~35–40 GB for 6.4B
- **NVLink**: Same allreduce heartbeat as T1 (DDP still syncs every step)
- **Temperature**: Potentially higher (more sustained compute)
- **Key question**: With the 6.4B config, memory usage (~35–40 GB) overlaps with what a large inference model might use. Can other signals still distinguish it?

## Hardware notes
- Works on both A100 and H100 with no changes
- The 6.4B config should fit on both (80GB, with checkpointing reducing activation memory by ~60%)

## Launch
```bash
torchrun --nproc_per_node=8 workloads/train_t5.py
```

## Output
```
data/t5_telemetry.csv
```

## Dependencies
- Same as T1 + `torch.utils.checkpoint`

---

## Implementation notes (2026-03-31)

### What was built
`workloads/train_t5.py` — fork of T1 with gradient checkpointing on every transformer block.

### Actual param counts (verified on CPU)
| Config | d_model | n_layers | n_heads | Actual params |
|---|---|---|---|---|
| Primary (6.4B) | 4096 | 32 | 32 | **6.71B** |
| Fallback (3.2B) | 3072 | 28 | 24 | **3.37B** (matches T1) |

### Key design decisions
- **Automatic OOM fallback**: tries 6.4B first with a full test forward+backward pass to trigger any OOM early, then catches `torch.cuda.OutOfMemoryError` and falls back to 3.2B. Cleanup (`del model, optimizer; torch.cuda.empty_cache()`) happens before fallback allocation.
- **`use_reentrant=False`**: uses the newer, safer checkpointing API that doesn't require `torch.autograd.Function` and handles non-tensor args correctly.
- **Checkpointing scope**: applied per transformer block (not per attention/FFN sublayer). This is the standard granularity — each block's activations are discarded and recomputed during backward.
- **Everything else identical to T1**: DDP, synthetic data, bf16, AdamW, batch=4, seq=2048, 5min duration, 30s warmup, same telemetry collector.

### Validation performed (no GPU execution)
- Syntax parse: OK
- Module import: OK
- CPU instantiation of both configs: OK, param counts verified
- CPU forward+backward with checkpointing on a tiny model: OK, all params received gradients

### Not yet run
The script has not been executed with `torchrun` — no GPU resources consumed. The 6.4B config's actual memory usage with checkpointing is still an estimate (~35-40 GB). If it still OOMs (unlikely but possible), the fallback to 3.2B will activate automatically.

### Launch command
```bash
torchrun --nproc_per_node=8 workloads/train_t5.py
```

### Output
```
data/t5_telemetry.csv
```

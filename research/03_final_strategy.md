# Final Strategy: Beat the SOTA

## Decision Summary

Stop inventing architecture. Start measuring and tuning. The path to winning:
1. **TTT done right** — correct optimizer, LR, layers, epochs within wall-clock budget
2. **Don't waste bits** — close the quantization entropy gap
3. **More context at eval time** — free BPB from longer attention windows

---

## Tier 1: Must-Do (Days 1-4)

### T1-A: Pre-Quant TTT + Aggressive HP Search
**Expected: 0.005-0.015 BPB | Effort: MEDIUM**

1. Fork best submission as baseline. Lock eval harness with config hashing.
2. Implement pre-quant TTT (adapt EMA model on val data before GPTQ).
3. Sweep:
   - Optimizer: Adam (lr=1e-4, 3e-4, 1e-3) vs SGD (lr=0.01, 0.03, 0.1) + momentum 0.9
   - TTT epochs: 1, 2, 3, 5
   - Chunk size: 256, 512, 1024, 2048
   - Layers to adapt: all, last-6, last-3, MLP-only, attention-only
4. Use reduced eval set (first 10 shards) for fast iteration.
5. **Gate (Day 2):** If Adam > SGD by >0.002, commit to Adam. If LoRA enables more epochs and wins, commit to LoRA.

### T1-B: LoRA TTT as SGD Replacement
**Expected: 0.001-0.004 BPB | Effort: LOW-MEDIUM**

1. Implement rank-4 and rank-8 LoRA adapters.
2. Benchmark: full-param TTT for N epochs vs LoRA TTT for 2N epochs (same wall-clock).
3. Metric: **BPB per second of eval time**, not BPB per epoch.
4. If LoRA wins, it *replaces* full-param TTT entirely.
5. **Gate (Day 3):** Binary go/no-go.

### T1-C: Quantization Compression Audit
**Expected: 0.002-0.006 BPB | Effort: LOW**

1. Measure actual entropy gap: theoretical entropy vs achieved Brotli size.
2. If gap > 0.5%: try byte-shuffle + zstd.
3. If gap > 1%: implement rANS coding.
4. Try reordering weight groups by variance before compression.
5. **Gate (Day 2):** If gap < 0.3%, stop. Brotli is good enough.

---

## Tier 2: Should-Do (Days 4-6)

### T2-A: Extended Eval Context (YaRN/NTK-aware RoPE)
**Expected: 0.003-0.010 BPB | Effort: MEDIUM**

Extend from 2048 to 4096/8192 context at eval time via NTK-aware RoPE scaling.
No retraining needed. Combine with TTT for more adaptation signal.
Verify stays within 10-min eval budget.

### T2-B: Per-Group Adaptive Bit Allocation
**Expected: 0.001-0.003 BPB | Effort: LOW**

Profile group sensitivity → int8 top-10%, int4 rest, int2 least sensitive.
Composes with T1-C compression improvements.

### T2-C: EMA Schedule + Training HP Sweep
**Expected: 0.001-0.003 BPB | Effort: LOW**

Test cosine EMA decay, batch size variations, LR warmup lengths.
Submit as overnight sweep jobs.

---

## Tier 3: Nice-to-Have (Days 7-8)

### T3-A: Training Data Filtering
Score FineWeb shards by perplexity, remove worst 10%. Expected: 0.001-0.002 BPB.

### T3-B: Distillation from Larger Teacher
Train 64MB model unconstrained, distill to 16MB. Only if ahead of schedule.
Expected: 0.002-0.005 BPB.

---

## Explicitly Rejected

| Proposal | Reason |
|---|---|
| Meta-Learning TTT | FA3 no 2nd-order grads; months of work for marginal gain |
| CaseOps Tokenizer | Invalidates all HP tuning; disqualification risk |
| RDT + LTI Stability | Current recurrence works; academic overhead |
| Differential Attention | 2x attention cost via FA3; QK-Gain already solves this |
| Value Residuals | Tried and removed by multiple competitors |
| Curriculum Learning | Single-epoch regime; below noise floor |
| Hybrid SSM | PR #1757 showed collapse at this scale |

---

## Implementation Schedule

| Day | Focus | Gate |
|---|---|---|
| Day 1 (Apr 21) | Baseline lock + TTT infra + entropy gap measurement | — |
| Day 2 (Apr 22) | TTT HP sweep + compression experiments | Drop T1-C if gap < 0.3% |
| Day 3 (Apr 23) | LoRA TTT comparison | Drop LoRA if it loses |
| Day 4 (Apr 24) | Full eval with best TTT + start context extension | Reassess if BPB > 1.030 |
| Day 5 (Apr 25) | Context window tuning + adaptive bit allocation | — |
| Day 6 (Apr 26) | EMA/training sweeps + integration | If < 1.025, polish. If > 1.028, escalate |
| Day 7 (Apr 27) | Data filtering OR distillation + integration | — |
| Day 8 (Apr 28) | Full eval, ablations, reproducibility (3 seeds) | — |
| Day 9 (Apr 29) | Submission prep, documentation, PR | Deadline buffer |

---

## Risk Mitigation

- **TTT doesn't help:** Debug weight norms before/after. Fall back to quant gap + context extension.
- **Can't beat 1.028:** Submit anyway — still beats merged SOTA (1.081) by 0.053.
- **Limited GPU access:** Prioritize TTT tuning (eval-time) over retraining.
- **Submission rejected:** Maintain "safe" branch with only well-established techniques.

---

## Target BPB

| Scenario | BPB | Confidence |
|---|---|---|
| Floor (Tier 1 only) | 1.026-1.028 | 70% |
| **Target (Tier 1 + 2)** | **1.023-1.026** | **50%** |
| Stretch (everything works) | 1.020-1.023 | 20% |
| Merged SOTA to beat | 1.0810 | Already beaten |

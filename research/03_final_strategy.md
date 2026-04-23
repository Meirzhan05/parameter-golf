# Final Strategy: Beat the SOTA (Legal Only)

## Decision Summary

Focus on legal improvements only. Pre-quant TTT violates Issue #1017 Conditions 1 & 3
and has been removed. All techniques below are unambiguously legal under Track A or Track B.

---

## ILLEGAL (Removed)

| Technique | Why Illegal |
|---|---|
| Pre-quant TTT (full val epochs before GPTQ) | Violates Condition 3: scores tokens after training on them |
| Pre-quant LoRA TTT | Same violation as above |

---

## Legal Improvements (New Tier 1)

### T1-A: Post-Quant Score-First TTT HP Tuning
**Expected: 0.002-0.008 BPB | Effort: LOW**

The merged SOTA already has legal score-first TTT (SGD, lr=0.005, 3 epochs per 32K chunk).
Tune it harder:
- Optimizer: try Adam instead of SGD
- LR: sweep 0.001, 0.003, 0.005, 0.01
- Epochs per chunk: 1, 3, 5
- Chunk size: 16K, 32K, 64K

### T1-B: Extended Eval Context (YaRN/NTK-aware RoPE)
**Expected: 0.003-0.010 BPB | Effort: MEDIUM**

Extend eval sliding window from 2048 to 4096 tokens via NTK-aware RoPE scaling.
No retraining needed. Purely eval-time, fully legal.

### T1-C: Per-Group Adaptive Bit Allocation
**Expected: 0.001-0.003 BPB | Effort: LOW**

Use Hessian traces to assign int8 to sensitive early layers, int5 to late layers.
Pure post-training change. Fully legal.

---

## Tier 2: Should-Do

### T2-A: EMA Schedule Optimization
**Expected: 0.001-0.003 BPB | Effort: LOW**

Test cosine EMA decay, batch size variations, LR warmup lengths.

### T2-B: Compression Audit
**Expected: 0.001-0.004 BPB | Effort: LOW**

Measure entropy gap. Try zstd vs Brotli. Byte-shuffle optimizations.

### T2-C: Training Data Filtering
**Expected: 0.001-0.002 BPB | Effort: LOW**

Score FineWeb shards by perplexity, remove worst 10%.

---

## Target BPB (Legal Only)

| Scenario | BPB | Confidence |
|---|---|---|
| Current merged SOTA | 1.0810 | baseline |
| With TTT HP tuning | 1.075-1.079 | 60% |
| + Extended context | 1.070-1.076 | 40% |
| + Adaptive bit allocation | 1.068-1.074 | 30% |

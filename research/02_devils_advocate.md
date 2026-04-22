# Devil's Advocate Review

## Verdicts on Researcher's 10 Proposals

| # | Proposal | Verdict | Realistic BPB | Key Objection |
|---|----------|---------|---------------|---------------|
| 1 | Meta-Learning TTT | **SKIP** | 0.001-0.005 | FA3 has no 2nd-order grad support; enormous effort |
| 2 | rANS Entropy Coding | **MODIFY** | 0.001-0.004 | Measure entropy gap first; Brotli already good |
| 3 | CaseOps Tokenizer | **SKIP** | 0.001-0.003 | Invalidates all HP tuning; scrutiny/disqualification risk |
| 4 | RDT + LTI Stability | **SKIP** | 0.000-0.003 | Current recurrence works; principled version = academic overhead |
| 5 | Differential Attention | **SKIP** | 0.000-0.002 | FA3 incompatible (2x cost); QK-Gain already solves this |
| 6 | Value Residuals | **SKIP** | 0.000-0.001 | Already tried and REMOVED by multiple competitors |
| 7 | Adaptive Bit Allocation | **PURSUE** | 0.001-0.003 | Sound, but limited headroom (int4 MLP already exists) |
| 8 | LoRA TTT | **MODIFY** | 0.001-0.004 | Replace TTT, don't layer. Test wall-clock efficiency |
| 9 | Curriculum Learning | **SKIP** | 0.000-0.001 | Single-epoch regime; below noise floor |
| 10 | Hybrid SSM | **SKIP** | negative likely | PR #1757 already showed collapse at this scale |

## Missed Opportunities (Not in Researcher's List)

### A. TTT Hyperparameter Tuning (0.002-0.008 BPB, LOW effort)
Try Adam instead of SGD, different LR schedules, chunk sizes (16K/32K/64K), which layers to adapt.

### B. Longer Eval-Time Context Window (0.003-0.010 BPB, MEDIUM effort)
Extend from 2048 to 4096/8192 with YaRN/NTK-aware RoPE. Purely eval-time change.

### C. Distillation from Larger Teacher (0.003-0.008 BPB, HIGH effort)
Train 64MB model without constraints, distill to 16MB student.

### D. EMA Schedule Optimization (0.001-0.002 BPB, LOW effort)
Non-constant EMA decay (start 0.999, end 0.9999).

### E. Training Data Filtering (0.001-0.002 BPB, LOW effort)
Remove worst 10% of FineWeb shards by perplexity.

### F. Batch Size Tuning (0.001-0.002 BPB, LOW effort)

## Core Thesis
> "The real gap is pre-quant TTT and TTT tuning — an engineering + hyperparameter problem, not an architecture problem. The researcher's proposals skew heavily toward architecture when the returns are in eval-time adaptation and quantization engineering."

## The Actual Bottleneck at ~1.028 BPB
1. **Quantization gap** (~0.012 BPP pre-quant to post-quant)
2. **Model capacity** (need ~2x more effective params for next 0.01 BPP)
3. **Training convergence** (4500 steps in 10min is not enough to fully converge)

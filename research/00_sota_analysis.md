# SOTA Analysis Report

## Competition: Parameter Golf (OpenAI Model Craft Challenge)
- **Objective**: Best language model in 16MB artifact, trained in <10min on 8×H100s
- **Metric**: Bits-per-byte (BPB) on FineWeb validation set
- **Deadline**: April 30, 2026

## Current SOTA

### Merged Leaderboard: 1.0810 BPB (bigbag, Apr 9)
- SP8192 tokenizer, 11 physical layers, 3-layer depth recurrence (17 virtual layers)
- 512d hidden, 8 heads / 4 KV heads (GQA), 4× MLP (2048)
- LeakyReLU(0.5)², Partial RoPE (16/64 dims), RMSNorm, Logit softcap 30.0
- Parallel residuals (layers 7+), U-Net skip gates
- MuonEq-R optimizer, Linear warmdown (72%), WD=0.095, EMA=0.9965
- int6 GPTQ + SDClip + Brotli-11, int8 embeddings
- Score-first SGD TTT (lr=0.005, momentum=0.9, 3 epochs/32K chunk)
- Artifact: 15,991,930 bytes (7,306 bytes margin)

### Actual Frontier (Open PRs): ~1.02840 BPB
| PR | BPB | Key Innovation |
|----|-----|---------------|
| #1758 | 1.02840 | Pre-Quant TTT LR=1e-3, unfrozen all blocks |
| #1738 | 1.03540 | CaseOps Tokenizer V15 + Pre-Quant AdamW TTT |
| #1735 | 1.04290 | 8-GPU Parallel Pre-Quant AdamW TTT (21 epochs) |
| #1769 | 1.06453 | CaseOps + GatedAttn + QuantGate + Loop4-5 + PhasedTTT |
| #1767 | 1.07209 | Alpha-Scaled LoRA + Warm-start A + WD 1.0 |

**Key insight**: The ~0.053 BPB gap from merged SOTA to frontier is almost entirely from **pre-quant TTT** — adapting the full-precision model on validation data before GPTQ quantization.

## Baseline → SOTA Evolution
- Baseline: 1.2244 BPB (9L, 512d, 1024 vocab)
- SP8192 vocab: ~0.02 BPB gain
- Depth recurrence: ~0.01 BPB gain
- Parallel residuals + skip gates: ~0.005 BPB gain
- GPTQ + SDClip + Brotli: ~0.005 BPB gain
- QK-Gain tuning: ~0.002 BPB gain
- Score-first TTT: ~0.003 BPB gain
- Pre-quant TTT: ~0.04-0.05 BPB gain (biggest single unlock)

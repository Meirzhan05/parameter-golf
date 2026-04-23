# MTP (Multi-Token Prediction) Experiment Results

## What is MTP?
Auxiliary prediction heads that predict 2-4 tokens ahead during training, giving the transformer richer gradient signals per step. Heads are discarded before GPTQ quantization — zero final artifact cost.

- **Paper**: "Better & Faster LLMs via Multi-token Prediction" — Meta, Apr 2024 (arxiv.org/abs/2404.19737)
- **Legal status**: ✓ Fully legal. Training-time only.

## Implementation
- Branch: `feature/mtp`
- File: `train_gpt_mtp.py` (decoded SOTA + MTP additions)
- Env vars: `MTP_NUM_HEADS` (default 2), `MTP_LOSS_WEIGHT` (default 0.2)
- Each MTP head k predicts token at position t+k+2 from position t's hidden state
- Heads: CastedLinear(model_dim=512, vocab_size=8192), zero-initialized
- Loss: weighted average of auxiliary cross-entropy losses added to main loss
- Excluded from: EMA, serialization, GPTQ quantization

## Results

| Config | Steps | Pre-quant BPB | Quantized BPB | Sliding BPB |
|---|---|---|---|---|
| **SOTA baseline (no MTP)** | **4581** | **1.0866** | **1.0793** | **~1.083** |
| MTP_NUM_HEADS=2, WEIGHT=0.2 | 4099 | 1.0989 | 1.1112 | — |
| MTP_NUM_HEADS=1, WEIGHT=0.2 | 4303 | 1.0963 | 1.1094 | 1.0926 |

## Analysis

### Why MTP hurt performance:
1. **Training step overhead**: Each MTP head adds a 512×8192 matmul per forward pass. With 2 heads, this costs ~10.5% of training steps (4099 vs 4581). With 1 head, ~6% (4303 vs 4581).
2. **Wall-clock is the binding constraint**: In this competition, training is capped at 10 minutes. Every millisecond per step directly translates to fewer total steps. MTP's auxiliary gradient benefit works when you have unlimited compute, but here fewer steps = worse model.
3. **Auxiliary loss didn't compensate**: The richer gradient signal from MTP was not enough to offset the lost training steps. The pre-quant BPB with MTP=1 (1.0963) is worse than baseline (1.0866) despite MTP providing multi-horizon gradient information.
4. **Code size issue**: The decoded train_gpt_mtp.py is 50KB vs the LZMA-compressed SOTA at ~8KB. This pushes the total artifact over the 16MB limit (16,026,346 bytes). Would need LZMA compression to be submission-valid.

### What would need to change for MTP to work:
- **Smaller MTP heads**: Project hidden states to a small dimension (e.g., 64) before projecting to vocab. Reduces 512×8192 matmul to 512×64 + 64×8192, which is much faster.
- **MTP only in early training**: Enable MTP for the first 50% of steps when the model benefits most from multi-horizon signals, then disable for the rest to recover step speed.
- **Faster hardware/longer time budget**: MTP's value proposition is better sample efficiency per step. With a longer training budget, the per-step overhead matters less.

## Conclusion
MTP is a net negative in the Parameter Golf setting. The 10-minute wall-clock constraint makes per-step overhead the dominant factor. Recommend moving to techniques that don't cost training steps (FOEM quantization, Newton-Muon optimizer, LoRA on recurrence).

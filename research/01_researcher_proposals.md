# Researcher Proposals: 10 Approaches to Beat SOTA

## Ranked by Expected Impact

### 1. Improved Pre-Quant TTT with Meta-Learning (0.005-0.015 BPB)
Train model initialization to be better at TTT adaptation via outer meta-learning loop (TTT-E2E).
Requires second-order gradients through inner loop; FA3 incompatible.

### 2. Rate-Constrained Quantization with rANS Entropy Coding (0.003-0.010 BPB)
Replace GPTQ+Brotli with rate-distortion optimal non-uniform quantization + rANS.
Could free 0.5-1.5MB for more capacity or finer quantization.

### 3. CaseOps Tokenizer + Further Optimization (0.002-0.008 BPB)
Retrain SentencePiece on CaseOps-preprocessed corpus. Extend to number factorization.
Already partially deployed in frontier submissions (PR #1729, #1738).

### 4. Recurrent-Depth Transformer with LTI Stability (0.005-0.012 BPB)
Replace ad-hoc depth recurrence with principled RDT + LTI stability constraints.
Allow 5-8x recurrence iterations (up from 2-3x). Based on OpenMythos.

### 5. Differential Attention (0.003-0.008 BPB)
Microsoft's Diff Transformer: attention as difference of two softmax maps.
~20-line change, no extra parameters. ICLR 2025.

### 6. Value Residual Connections / ResFormer (0.002-0.005 BPB)
Cache first-layer values and add them (scaled) to all subsequent layers.
~10 lines of code, one scalar parameter per layer.

### 7. Per-Group Adaptive Bit Allocation in GPTQ (0.002-0.005 BPB)
Use Hessian traces to assign int7/8 to sensitive early layers, int5 to late layers.
Stable cross-seed (r=0.997 per PR #1412). Pure post-training change.

### 8. LoRA-Based TTT with Warm-Start + Alpha Scaling (0.003-0.008 BPB)
Layer LoRA-TTT on top of pre-quant TTT for per-document adaptation.
PR #1767 shows 1.07209 BPB with LoRA alone.

### 9. Curriculum Learning (0.001-0.004 BPB)
Order training data easy-to-hard. Pre-compute difficulty scores offline.

### 10. Hybrid SSM-Attention (0.005-0.015 BPB)
Replace some attention layers with Mamba-2/3 SSM layers.
Note: PR #1757 already showed collapse at small scale.

## Recommended Stack Order
1. Start from PR #1758 stack (pre-quant TTT + all accumulated improvements)
2. Add Differential Attention (#5) — low-hanging fruit
3. Add Value Residual Connections (#6) — low-cost
4. Implement per-group bit allocation (#7)
5. Explore LTI-stabilized deeper recurrence (#4) if time permits

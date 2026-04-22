"""
Tier 1 Improvements for Parameter Golf
=======================================

This module extends the SOTA train_gpt.py with three Tier 1 improvements:

T1-A: Pre-Quantization TTT (adapt EMA model on val data BEFORE GPTQ)
T1-B: LoRA-based TTT (low-rank adapters for faster per-step TTT)
T1-C: Compression audit (measure entropy gap, try alternative compressors)

Usage:
    Import and call the functions from the main training script, or run
    standalone for compression audit.

All features are controlled via environment variables for easy HP sweeping.
"""

import copy
import io
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# T1-A: Pre-Quantization Test-Time Training
# =============================================================================
# The key insight: adapt the full-precision EMA model on validation data
# BEFORE quantization. The full-precision model has much more capacity to
# learn from val data than a quantized one.

class PreQuantTTTConfig:
    """Configuration for pre-quantization TTT, all from env vars."""
    enabled = bool(int(os.environ.get('PREQUANT_TTT_ENABLED', '0')))
    optimizer = os.environ.get('PREQUANT_TTT_OPTIMIZER', 'adamw')  # 'sgd' or 'adamw'
    lr = float(os.environ.get('PREQUANT_TTT_LR', 1e-3))
    epochs = int(os.environ.get('PREQUANT_TTT_EPOCHS', 5))
    momentum = float(os.environ.get('PREQUANT_TTT_MOMENTUM', 0.9))
    weight_decay = float(os.environ.get('PREQUANT_TTT_WD', 0.01))
    beta1 = float(os.environ.get('PREQUANT_TTT_BETA1', 0.9))
    beta2 = float(os.environ.get('PREQUANT_TTT_BETA2', 0.999))
    grad_clip = float(os.environ.get('PREQUANT_TTT_GRAD_CLIP', 1.0))
    seq_len = int(os.environ.get('PREQUANT_TTT_SEQ_LEN', 1024))
    batch_seqs = int(os.environ.get('PREQUANT_TTT_BATCH_SEQS', 32))
    # Which layers to adapt: 'all', 'last_N' (e.g. 'last_6'), 'mlp_only', 'attn_only'
    layer_selection = os.environ.get('PREQUANT_TTT_LAYERS', 'all')
    # Cosine LR decay over epochs
    cosine_decay = bool(int(os.environ.get('PREQUANT_TTT_COSINE_DECAY', '1')))
    # Use LoRA instead of full-param (delegates to T1-B)
    use_lora = bool(int(os.environ.get('PREQUANT_TTT_USE_LORA', '0')))
    lora_rank = int(os.environ.get('PREQUANT_TTT_LORA_RANK', 8))


def _select_params_for_ttt(model, layer_selection):
    """Select which parameters to adapt based on layer_selection config.

    Returns:
        list of (name, param) tuples that should have requires_grad=True
    """
    all_params = list(model.named_parameters())

    if layer_selection == 'all':
        return all_params

    if layer_selection.startswith('last_'):
        n = int(layer_selection.split('_')[1])
        num_blocks = len(model.blocks)
        target_block_indices = set(range(max(0, num_blocks - n), num_blocks))
        selected = []
        for name, param in all_params:
            if 'blocks.' in name:
                block_idx = int(name.split('blocks.')[1].split('.')[0])
                if block_idx in target_block_indices:
                    selected.append((name, param))
            # Always include final norm and head
            elif 'final_norm' in name or 'lm_head' in name or 'tok_emb' in name:
                selected.append((name, param))
        return selected

    if layer_selection == 'mlp_only':
        return [(n, p) for n, p in all_params if '.mlp.' in n]

    if layer_selection == 'attn_only':
        return [(n, p) for n, p in all_params if '.attn.' in n]

    # Default: all
    return all_params


def pre_quant_ttt(model, val_data, config, device, log_fn=print):
    """Adapt the full-precision EMA model on validation data before quantization.

    This is the single highest-impact technique in the competition (~0.04-0.05 BPB).

    Args:
        model: The full-precision EMA model (not quantized, not compiled)
        val_data: ValidationData object with val_tokens
        config: PreQuantTTTConfig
        device: torch device
        log_fn: logging function

    Returns:
        model: The adapted model (modified in-place)
    """
    if not config.enabled:
        return model

    log_fn(f"pre_quant_ttt: starting optimizer={config.optimizer} lr={config.lr} "
           f"epochs={config.epochs} layers={config.layer_selection}")

    t0 = time.perf_counter()

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad_(False)

    # Select and unfreeze target parameters
    selected = _select_params_for_ttt(model, config.layer_selection)
    ttt_params = []
    for name, param in selected:
        param.requires_grad_(True)
        ttt_params.append(param)

    log_fn(f"pre_quant_ttt: adapting {len(ttt_params)} parameter groups, "
           f"{sum(p.numel() for p in ttt_params)} params")

    # Build optimizer
    if config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            ttt_params, lr=config.lr,
            momentum=config.momentum, weight_decay=config.weight_decay
        )
    elif config.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            ttt_params, lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            ttt_params, lr=config.lr,
            betas=(config.beta1, config.beta2)
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    # Prepare validation data as training sequences
    val_tokens = val_data.val_tokens.to(device)
    total_tokens = val_tokens.numel() - 1
    seq_len = config.seq_len
    num_seqs = total_tokens // seq_len
    batch_seqs = config.batch_seqs

    # Distributed: split sequences across ranks
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    my_seq_start = num_seqs * rank // world_size
    my_seq_end = num_seqs * (rank + 1) // world_size
    my_num_seqs = my_seq_end - my_seq_start

    model.train()

    for epoch in range(config.epochs):
        # Cosine LR decay across epochs
        if config.cosine_decay and config.epochs > 1:
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * epoch / max(config.epochs - 1, 1)))
            for pg in optimizer.param_groups:
                pg['lr'] = config.lr * lr_scale

        epoch_loss = 0.0
        num_batches = 0

        # Shuffle sequence order each epoch
        perm = torch.randperm(my_num_seqs)

        for bi in range(0, my_num_seqs, batch_seqs):
            be = min(bi + batch_seqs, my_num_seqs)
            actual_batch = be - bi

            # Gather sequences
            x_batch = torch.zeros(actual_batch, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(actual_batch, seq_len, dtype=torch.int64, device=device)

            for i in range(actual_batch):
                seq_idx = my_seq_start + perm[bi + i].item()
                start = seq_idx * seq_len
                end = start + seq_len + 1
                # num_seqs guarantees end <= val_tokens.numel()
                chunk = val_tokens[start:end].to(torch.int64)
                x_batch[i] = chunk[:-1]
                y_batch[i] = chunk[1:]

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = model(x_batch, y_batch)

            loss.backward()

            # Gradient sync across GPUs
            if world_size > 1:
                for p in ttt_params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(ttt_params, config.grad_clip)

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        current_lr = optimizer.param_groups[0]['lr']
        log_fn(f"pre_quant_ttt: epoch {epoch+1}/{config.epochs} "
               f"loss={avg_loss:.4f} lr={current_lr:.6f}")

    # Re-enable all gradients for potential downstream use
    for p in model.parameters():
        p.requires_grad_(True)

    elapsed = time.perf_counter() - t0
    log_fn(f"pre_quant_ttt: completed in {elapsed:.1f}s")

    model.eval()
    return model


# =============================================================================
# T1-B: LoRA-based TTT
# =============================================================================
# LoRA adapters for faster per-step TTT. Each step is ~3-5x faster than
# full-param, enabling more epochs in the same wall-clock budget.

class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for a linear layer.

    Wraps an existing nn.Linear (or CastedLinear) with a low-rank
    A*B adapter: output = original(x) + x @ A @ B * (alpha/rank)

    Only A and B are trainable. The original weight is frozen.
    """

    def __init__(self, original_linear, rank=8, alpha=16.0):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_linear.weight.shape[1]
        out_features = original_linear.weight.shape[0]

        # A: input projection (in_features -> rank), initialized with Kaiming
        # Use bfloat16 to match autocast dtype and avoid memory explosion
        self.lora_A = nn.Parameter(
            torch.empty(in_features, rank, device=original_linear.weight.device,
                        dtype=torch.bfloat16)
        )
        # Kaiming init in float32 then cast down for precision
        with torch.no_grad():
            tmp = torch.empty_like(self.lora_A, dtype=torch.float32)
            nn.init.kaiming_uniform_(tmp, a=math.sqrt(5))
            self.lora_A.data.copy_(tmp.to(torch.bfloat16))

        # B: output projection (rank -> out_features), initialized to zero
        # so LoRA starts as identity
        self.lora_B = nn.Parameter(
            torch.zeros(rank, out_features, device=original_linear.weight.device,
                        dtype=torch.bfloat16)
        )

        # Freeze original
        for p in self.original.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        # Original forward (frozen)
        base_out = self.original(x)
        # LoRA delta — keep in same dtype as x to avoid memory explosion
        # and to benefit from autocast (bfloat16 on H100)
        lora_out = (x @ self.lora_A.to(x.dtype) @ self.lora_B.to(x.dtype)) * self.scaling
        return base_out + lora_out

    def merge_and_remove(self):
        """Merge LoRA weights into the original linear and return it."""
        with torch.no_grad():
            # Compute in float32 for precision, then cast to weight dtype
            A = self.lora_A.float()
            B = self.lora_B.float()
            delta = (A @ B * self.scaling).T  # (out, in)
            self.original.weight.data += delta.to(self.original.weight.dtype)
        return self.original


def apply_lora_to_model(model, rank=8, alpha=16.0, target_modules=None):
    """Apply LoRA adapters to linear layers in the model.

    Args:
        model: The GPT model
        rank: LoRA rank
        alpha: LoRA alpha (scaling = alpha/rank)
        target_modules: List of substrings to match. Default targets Q/K/V/proj.

    Returns:
        lora_params: List of LoRA parameters (for the optimizer)
        lora_modules: Dict mapping name -> LoRALinear (for merge later)
    """
    if target_modules is None:
        target_modules = ['c_q', 'c_k', 'c_v', 'proj', 'fc']

    lora_params = []
    lora_modules = {}

    for block_idx, block in enumerate(model.blocks):
        for name, module in block.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            # Check if this module should get LoRA
            if not any(t in name for t in target_modules):
                continue

            full_name = f"blocks.{block_idx}.{name}"
            lora_wrapper = LoRALinear(module, rank=rank, alpha=alpha)
            lora_modules[full_name] = lora_wrapper

            # Replace the module in the parent
            parts = name.split('.')
            parent = block
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], lora_wrapper)

            lora_params.extend([lora_wrapper.lora_A, lora_wrapper.lora_B])

    return lora_params, lora_modules


def merge_lora_into_model(model, lora_modules):
    """Merge all LoRA adapters back into the base model weights.

    This must be called before quantization so the LoRA adaptations
    are baked into the weights that GPTQ will quantize.
    """
    for full_name, lora_wrapper in lora_modules.items():
        # Parse "blocks.N.attn.c_q" -> navigate to parent
        parts = full_name.split('.')
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        # Replace LoRA wrapper with merged original
        merged = lora_wrapper.merge_and_remove()
        setattr(parent, parts[-1], merged)


def pre_quant_ttt_lora(model, val_data, config, device, log_fn=print):
    """Pre-quant TTT using LoRA adapters for faster adaptation.

    Same concept as pre_quant_ttt but only updates low-rank adapters,
    enabling ~3-5x more steps in the same wall-clock.

    After adaptation, LoRA weights are merged back into the base model
    so GPTQ quantizes the adapted weights.
    """
    if not config.enabled or not config.use_lora:
        return model

    log_fn(f"pre_quant_ttt_lora: starting rank={config.lora_rank} "
           f"optimizer={config.optimizer} lr={config.lr} epochs={config.epochs}")

    t0 = time.perf_counter()

    # Freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    # Apply LoRA adapters
    lora_params, lora_modules = apply_lora_to_model(
        model, rank=config.lora_rank, alpha=config.lora_rank * 2.0
    )

    log_fn(f"pre_quant_ttt_lora: applied {len(lora_modules)} LoRA adapters, "
           f"{sum(p.numel() for p in lora_params)} trainable params")

    # Build optimizer (only LoRA params)
    if config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            lora_params, lr=config.lr,
            momentum=config.momentum, weight_decay=config.weight_decay
        )
    elif config.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            lora_params, lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            lora_params, lr=config.lr,
            betas=(config.beta1, config.beta2)
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    # Prepare validation data
    val_tokens = val_data.val_tokens.to(device)
    total_tokens = val_tokens.numel() - 1
    seq_len = config.seq_len
    num_seqs = total_tokens // seq_len
    batch_seqs = config.batch_seqs

    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    my_seq_start = num_seqs * rank // world_size
    my_seq_end = num_seqs * (rank + 1) // world_size
    my_num_seqs = my_seq_end - my_seq_start

    model.train()

    for epoch in range(config.epochs):
        if config.cosine_decay and config.epochs > 1:
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * epoch / max(config.epochs - 1, 1)))
            for pg in optimizer.param_groups:
                pg['lr'] = config.lr * lr_scale

        epoch_loss = 0.0
        num_batches = 0
        perm = torch.randperm(my_num_seqs)

        for bi in range(0, my_num_seqs, batch_seqs):
            be = min(bi + batch_seqs, my_num_seqs)
            actual_batch = be - bi

            x_batch = torch.zeros(actual_batch, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(actual_batch, seq_len, dtype=torch.int64, device=device)

            for i in range(actual_batch):
                seq_idx = my_seq_start + perm[bi + i].item()
                start = seq_idx * seq_len
                end = start + seq_len + 1
                # num_seqs guarantees end <= val_tokens.numel()
                chunk = val_tokens[start:end].to(torch.int64)
                x_batch[i] = chunk[:-1]
                y_batch[i] = chunk[1:]

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = model(x_batch, y_batch)

            loss.backward()

            if world_size > 1:
                for p in lora_params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(lora_params, config.grad_clip)

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        current_lr = optimizer.param_groups[0]['lr']
        log_fn(f"pre_quant_ttt_lora: epoch {epoch+1}/{config.epochs} "
               f"loss={avg_loss:.4f} lr={current_lr:.6f}")

    # Merge LoRA weights back into base model
    merge_lora_into_model(model, lora_modules)
    log_fn("pre_quant_ttt_lora: merged LoRA weights into base model")

    # Re-enable all gradients
    for p in model.parameters():
        p.requires_grad_(True)

    elapsed = time.perf_counter() - t0
    log_fn(f"pre_quant_ttt_lora: completed in {elapsed:.1f}s")

    model.eval()
    return model


# =============================================================================
# T1-C: Compression Audit & Alternative Compressors
# =============================================================================

def compute_entropy(data_bytes):
    """Compute Shannon entropy in bits per byte of raw data."""
    arr = np.frombuffer(data_bytes, dtype=np.uint8)
    if len(arr) == 0:
        return 0.0
    counts = np.bincount(arr, minlength=256).astype(np.float64)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    return entropy


def compression_audit(quant_result, quant_meta, compressor='brotli', log_fn=print):
    """Measure the entropy gap between theoretical minimum and actual compression.

    This tells us how much room there is for improvement in compression.

    Args:
        quant_result: The quantized weight dict (matches SOTA's 'w' key)
        quant_meta: The quantization metadata dict (matches SOTA's 'm' key)
        compressor: Current compressor being used
        log_fn: logging function

    Returns:
        dict with audit results
    """
    log_fn("compression_audit: analyzing quantized weight distribution...")

    # Serialize in the same format as SOTA's serialize() function:
    # torch.save({'w': quant_result, 'm': quant_meta}, buf)
    buf = io.BytesIO()
    torch.save({'w': quant_result, 'm': quant_meta}, buf)
    raw_bytes = buf.getvalue()

    # Compute theoretical entropy
    entropy_bpb = compute_entropy(raw_bytes)
    theoretical_min_bytes = len(raw_bytes) * entropy_bpb / 8.0

    # Measure actual compression with current compressor
    compressed_sizes = {}

    # Brotli
    try:
        import brotli
        # With byte-shuffle (as in SOTA)
        shuffled = _byte_shuffle_for_audit(raw_bytes, stride=2)
        brotli_compressed = brotli.compress(shuffled, quality=11)
        compressed_sizes['brotli_q11_shuffled'] = len(brotli_compressed)
        # Without byte-shuffle
        brotli_plain = brotli.compress(raw_bytes, quality=11)
        compressed_sizes['brotli_q11_plain'] = len(brotli_plain)
    except ImportError:
        log_fn("compression_audit: brotli not available, skipping")

    # zstd (alternative)
    try:
        import zstandard as zstd
        compressor_zstd = zstd.ZstdCompressor(level=22)
        zstd_compressed = compressor_zstd.compress(raw_bytes)
        compressed_sizes['zstd_22_plain'] = len(zstd_compressed)
        # With byte-shuffle
        shuffled = _byte_shuffle_for_audit(raw_bytes, stride=2)
        zstd_shuffled = compressor_zstd.compress(shuffled)
        compressed_sizes['zstd_22_shuffled'] = len(zstd_shuffled)
    except ImportError:
        log_fn("compression_audit: zstandard not available, skipping")

    # LZMA
    import lzma
    lzma_compressed = lzma.compress(raw_bytes, preset=6)
    compressed_sizes['lzma_6'] = len(lzma_compressed)

    # zlib
    import zlib
    zlib_compressed = zlib.compress(raw_bytes, 9)
    compressed_sizes['zlib_9'] = len(zlib_compressed)

    # Report
    log_fn(f"compression_audit: raw size = {len(raw_bytes):,} bytes")
    log_fn(f"compression_audit: entropy = {entropy_bpb:.4f} bits/byte")
    log_fn(f"compression_audit: theoretical min = {theoretical_min_bytes:,.0f} bytes")

    best_name = None
    best_size = float('inf')

    for name, size in sorted(compressed_sizes.items(), key=lambda x: x[1]):
        gap_pct = (size - theoretical_min_bytes) / theoretical_min_bytes * 100
        log_fn(f"compression_audit:   {name}: {size:,} bytes "
               f"(gap: {gap_pct:.1f}% over theoretical)")
        if size < best_size:
            best_size = size
            best_name = name

    if best_name:
        current_best = compressed_sizes.get(f'{compressor}_q11_shuffled',
                                            compressed_sizes.get(f'{compressor}_q11_plain',
                                                                 float('inf')))
        savings = current_best - best_size
        log_fn(f"compression_audit: best = {best_name} ({best_size:,} bytes)")
        if savings > 0:
            log_fn(f"compression_audit: potential savings over current: "
                   f"{savings:,} bytes ({savings/current_best*100:.1f}%)")
        else:
            log_fn(f"compression_audit: current compressor is already optimal or near-optimal")

    return {
        'raw_bytes': len(raw_bytes),
        'entropy_bpb': entropy_bpb,
        'theoretical_min_bytes': theoretical_min_bytes,
        'compressed_sizes': compressed_sizes,
        'best_compressor': best_name,
        'best_size': best_size,
    }


def _byte_shuffle_for_audit(data, stride=2):
    """Byte-shuffle for compression audit (matches SOTA _byte_shuffle)."""
    if stride <= 1 or len(data) < stride:
        return data
    src = np.frombuffer(data, dtype=np.uint8)
    n = len(src)
    out = np.empty(n, dtype=np.uint8)
    dest_off = 0
    for pos in range(stride):
        chunk = src[pos::stride]
        out[dest_off:dest_off + len(chunk)] = chunk
        dest_off += len(chunk)
    return bytes(out)


# =============================================================================
# Integration: Modified train_and_eval flow
# =============================================================================

def integrate_tier1_into_train_and_eval(
    base_model, val_data, h, device, log_fn=print
):
    """Call this after training and EMA application, before serialize().

    This is the insertion point for pre-quant TTT in the main pipeline.

    Flow:
        1. Train model -> get EMA-applied base_model  (existing)
        2. >>> Pre-quant TTT (this function) <<<       (NEW - Tier 1)
        3. serialize() -> GPTQ quantize -> compress    (existing)

    Args:
        base_model: The full-precision EMA model
        val_data: ValidationData object
        h: Hyperparameters
        device: torch device
        log_fn: logging function

    Returns:
        base_model: The adapted model
    """
    config = PreQuantTTTConfig()

    if config.enabled:
        if config.use_lora:
            base_model = pre_quant_ttt_lora(
                base_model, val_data, config, device, log_fn
            )
        else:
            base_model = pre_quant_ttt(
                base_model, val_data, config, device, log_fn
            )

    return base_model


# =============================================================================
# Standalone test / usage example
# =============================================================================

if __name__ == '__main__':
    print("Tier 1 Improvements Module")
    print("=" * 50)
    print()
    print("T1-A: Pre-Quant TTT")
    print("  Enable: PREQUANT_TTT_ENABLED=1")
    print("  Optimizer: PREQUANT_TTT_OPTIMIZER=adamw (or sgd, adam)")
    print("  LR: PREQUANT_TTT_LR=1e-3")
    print("  Epochs: PREQUANT_TTT_EPOCHS=5")
    print("  Layers: PREQUANT_TTT_LAYERS=all (or last_6, mlp_only, attn_only)")
    print()
    print("T1-B: LoRA TTT")
    print("  Enable: PREQUANT_TTT_ENABLED=1 PREQUANT_TTT_USE_LORA=1")
    print("  Rank: PREQUANT_TTT_LORA_RANK=8")
    print()
    print("T1-C: Compression Audit")
    print("  Run compression_audit() on quantized state dict")
    print()

    # Quick LoRA sanity check
    print("Running LoRA sanity check...")
    linear = nn.Linear(64, 128, bias=False)
    lora = LoRALinear(linear, rank=4, alpha=8.0)
    x = torch.randn(2, 10, 64)
    out = lora(x)
    assert out.shape == (2, 10, 128), f"Expected (2, 10, 128), got {out.shape}"
    print(f"  LoRA output shape: {out.shape} OK")

    # Verify merge
    merged = lora.merge_and_remove()
    out2 = merged(x)
    assert out2.shape == (2, 10, 128)
    print(f"  LoRA merge shape: {out2.shape} OK")

    # Entropy computation check
    test_data = bytes(range(256)) * 100
    ent = compute_entropy(test_data)
    print(f"  Entropy of uniform bytes: {ent:.4f} bits/byte (expected: 8.0)")
    assert abs(ent - 8.0) < 0.01

    print()
    print("All sanity checks passed!")

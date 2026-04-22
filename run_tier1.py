"""
run_tier1.py — Ready-to-run script for Tier 1 improvements
===========================================================

This script wraps the SOTA train_gpt.py and injects pre-quant TTT
between EMA application and GPTQ quantization.

Usage:
    # Full run with pre-quant TTT (AdamW, lr=1e-3, 5 epochs, all layers)
    PREQUANT_TTT_ENABLED=1 \
    PREQUANT_TTT_OPTIMIZER=adamw \
    PREQUANT_TTT_LR=1e-3 \
    PREQUANT_TTT_EPOCHS=5 \
    torchrun --nproc_per_node=8 run_tier1.py

    # With LoRA TTT instead (faster per-step, more epochs possible)
    PREQUANT_TTT_ENABLED=1 \
    PREQUANT_TTT_USE_LORA=1 \
    PREQUANT_TTT_LORA_RANK=8 \
    PREQUANT_TTT_OPTIMIZER=adamw \
    PREQUANT_TTT_LR=1e-3 \
    PREQUANT_TTT_EPOCHS=10 \
    torchrun --nproc_per_node=8 run_tier1.py

    # Only adapt last 6 layers
    PREQUANT_TTT_ENABLED=1 \
    PREQUANT_TTT_LAYERS=last_6 \
    torchrun --nproc_per_node=8 run_tier1.py

    # Run without pre-quant TTT (baseline, same as original SOTA)
    torchrun --nproc_per_node=8 run_tier1.py

All SOTA env vars (SEED, DATA_DIR, etc.) still work as before.
"""

import os
import sys
import time

# ---------------------------------------------------------------------------
# Step 1: Decode and exec the SOTA train_gpt.py to get all its definitions
# into our namespace. We import everything it defines.
# ---------------------------------------------------------------------------

# The SOTA code is LZMA-compressed in records/. We decode it and exec it
# to get all class/function definitions, then override train_and_eval.
SOTA_PATH = os.environ.get(
    'SOTA_PATH',
    os.path.join(os.path.dirname(__file__),
                 'records/track_10min_16mb/'
                 '2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/'
                 'train_gpt.py')
)

# Read and decode the LZMA-compressed SOTA
import lzma as _L
import base64 as _B

with open(SOTA_PATH, 'r') as f:
    sota_source = f.read()

# Extract the b85-encoded blob
_blob_start = sota_source.index('b85decode("') + len('b85decode("')
_blob_end = sota_source.index('"', _blob_start)
_blob = sota_source[_blob_start:_blob_end]

_decoded = _L.decompress(
    _B.b85decode(_blob.encode()),
    format=_L.FORMAT_RAW,
    filters=[{"id": _L.FILTER_LZMA2}]
)
_sota_code = _decoded.decode('utf-8')

# Patch Python 3.12+ nested-quote f-strings for Python 3.11 compatibility.
# PEP 701 allows reusing the same quote type inside f-string braces, but
# Python 3.11 does not support this. Fix all instances.

# Line 266: log(f"  {cat}: {", ".join(sorted(categories[cat]))}")
_sota_code = _sota_code.replace(
    '''log(f"  {cat}: {", ".join(sorted(categories[cat]))}")''',
    '''log("  {}: {}".format(cat, ", ".join(sorted(categories[cat]))))'''
)

# Line 440: .glob("fineweb_train_*.bin") inside f-string
_sota_code = _sota_code.replace(
    '''.glob("fineweb_train_*.bin"))}''',
    """.glob('fineweb_train_*.bin'))}"""
)

# Execute the SOTA code in our module namespace so we get all definitions
# (GPT, Hyperparameters, train_model, serialize, eval_val, etc.)
exec(compile(_sota_code, SOTA_PATH, 'exec'), globals())

# ---------------------------------------------------------------------------
# Step 2: Import Tier 1 improvements
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from tier1_improvements import (
    PreQuantTTTConfig,
    pre_quant_ttt,
    pre_quant_ttt_lora,
    compression_audit,
)

# ---------------------------------------------------------------------------
# Step 3: Override train_and_eval with our patched version
# ---------------------------------------------------------------------------

# Save the original
_original_train_and_eval = train_and_eval  # noqa: F821 (defined by exec)


def train_and_eval(h, device):
    """Patched train_and_eval with Tier 1 pre-quant TTT."""
    import random
    import numpy as np
    import torch
    from pathlib import Path

    random.seed(h.seed)
    np.random.seed(h.seed)
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed_all(h.seed)

    val_data = ValidationData(h, device)  # noqa: F821
    log(f"train_shards: {len(list(Path(h.datasets_dir).resolve().glob('fineweb_train_*.bin')))}")  # noqa: F821
    log(f"val_tokens: {val_data.val_tokens.numel() - 1}")

    # ===== Step 1: Train the model (unchanged from SOTA) =====
    base_model, compiled_model = train_model(h, device, val_data)  # noqa: F821
    torch._dynamo.reset()

    # ===== Step 2: Evaluate pre-quant post-EMA baseline =====
    timed_eval('pre-quantization post-ema', eval_val, h, device, val_data, compiled_model)  # noqa: F821

    # ===== Step 3: PRE-QUANT TTT (Tier 1 insertion) =====
    config = PreQuantTTTConfig()
    if config.enabled:
        log("=" * 60)
        log("TIER 1: Pre-Quantization Test-Time Training")
        log("=" * 60)

        if config.use_lora:
            base_model = pre_quant_ttt_lora(
                base_model, val_data, config, device, log
            )
        else:
            base_model = pre_quant_ttt(
                base_model, val_data, config, device, log
            )

        # Re-compile and evaluate after TTT to measure improvement
        torch._dynamo.reset()
        compiled_model_post_ttt = torch.compile(
            base_model, dynamic=False, fullgraph=True
        )
        timed_eval(
            'pre-quantization post-ttt', eval_val,
            h, device, val_data, compiled_model_post_ttt
        )
        del compiled_model_post_ttt
        torch._dynamo.reset()

    # ===== Step 4: Serialize (GPTQ quantize + compress) =====
    serialize(h, base_model, Path(__file__).read_text(encoding='utf-8'))  # noqa: F821

    if h.distributed:
        import torch.distributed as dist
        dist.barrier()

    # ===== Step 5: Deserialize and evaluate quantized model =====
    eval_model = deserialize(h, device)  # noqa: F821
    if h.num_loops > 0:
        eval_model.looping_active = True
    compiled_model = torch.compile(eval_model, dynamic=False, fullgraph=True)
    timed_eval('quantized', eval_val, h, device, val_data, compiled_model)

    if h.sliding_window_enabled:
        timed_eval(
            'quantized_sliding_window', eval_val_sliding,  # noqa: F821
            h, device, val_data, eval_model
        )

    if h.ttt_enabled and h.sliding_window_enabled:
        del eval_model, compiled_model
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        ttt_model = deserialize(h, device)
        if h.num_loops > 0:
            ttt_model.looping_active = True
        timed_eval(
            'quantized_ttt', eval_val_ttt,  # noqa: F821
            h, device, val_data, ttt_model
        )
        del ttt_model

    if h.etlb_enabled and h.sliding_window_enabled:
        if 'eval_model' not in dir():
            eval_model = deserialize(h, device)
            if h.num_loops > 0:
                eval_model.looping_active = True
        timed_eval(
            'quantized_sliding_etlb', eval_val_sliding_etlb,  # noqa: F821
            h, device, val_data, eval_model
        )


# ---------------------------------------------------------------------------
# Step 4: Run main() — it will call our patched train_and_eval
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()  # noqa: F821 (defined by exec from SOTA code)

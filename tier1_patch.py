"""
Tier 1 Integration Patch
========================

This file shows the exact modifications needed to the SOTA train_gpt.py
to integrate all Tier 1 improvements. It patches the train_and_eval function.

Usage in the SOTA train_gpt.py:
    Replace the train_and_eval() function body with the patched version below.

The patch inserts pre-quant TTT between EMA application and GPTQ serialization.
"""

# The ORIGINAL train_and_eval flow (from SOTA, lines 439-454):
#
#   def train_and_eval(h, device):
#       ...
#       base_model, compiled_model = train_model(h, device, val_data)
#       torch._dynamo.reset()
#       timed_eval('pre-quantization post-ema', eval_val, h, device, val_data, compiled_model)
#       serialize(h, base_model, Path(__file__).read_text(encoding='utf-8'))
#       ...
#
# The PATCHED flow inserts pre-quant TTT after EMA eval, before serialize:
#
#   def train_and_eval(h, device):
#       ...
#       base_model, compiled_model = train_model(h, device, val_data)
#       torch._dynamo.reset()
#       timed_eval('pre-quantization post-ema', eval_val, h, device, val_data, compiled_model)
#
#       # >>> TIER 1 INSERTION POINT <<<
#       from tier1_improvements import integrate_tier1_into_train_and_eval
#       base_model = integrate_tier1_into_train_and_eval(
#           base_model, val_data, h, device, log
#       )
#       # Re-compile after TTT modified weights
#       torch._dynamo.reset()
#       compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
#       timed_eval('pre-quantization post-ttt', eval_val, h, device, val_data, compiled_model)
#
#       serialize(h, base_model, Path(__file__).read_text(encoding='utf-8'))
#       ...


def patched_train_and_eval(h, device):
    """Drop-in replacement for train_and_eval with Tier 1 improvements.

    Import and call this instead of the original train_and_eval.
    """
    import random
    import numpy as np
    import torch
    from pathlib import Path

    # These would be imported from the SOTA train_gpt.py
    # Shown here as documentation of what's needed:
    #   from train_gpt import (
    #       ValidationData, train_model, eval_val, eval_val_sliding,
    #       eval_val_ttt, serialize, deserialize, timed_eval, log,
    #       GPT, restore_fp32_params
    #   )
    from tier1_improvements import (
        integrate_tier1_into_train_and_eval,
        PreQuantTTTConfig,
        compression_audit,
    )

    random.seed(h.seed)
    np.random.seed(h.seed)
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed_all(h.seed)

    val_data = ValidationData(h, device)
    log(f"train_shards: {len(list(Path(h.datasets_dir).resolve().glob('fineweb_train_*.bin')))}")
    log(f"val_tokens: {val_data.val_tokens.numel() - 1}")

    # Step 1: Train the model (unchanged)
    base_model, compiled_model = train_model(h, device, val_data)
    torch._dynamo.reset()

    # Step 2: Evaluate pre-quant post-EMA baseline (unchanged)
    timed_eval('pre-quantization post-ema', eval_val, h, device, val_data, compiled_model)

    # Step 3: >>> TIER 1: Pre-Quant TTT <<< (NEW)
    base_model = integrate_tier1_into_train_and_eval(
        base_model, val_data, h, device, log
    )

    # Re-evaluate after TTT to measure improvement
    config = PreQuantTTTConfig()
    if config.enabled:
        torch._dynamo.reset()
        compiled_model_post_ttt = torch.compile(base_model, dynamic=False, fullgraph=True)
        timed_eval('pre-quantization post-ttt', eval_val, h, device, val_data, compiled_model_post_ttt)
        del compiled_model_post_ttt

    # Step 4: Serialize (GPTQ quantize + compress) (unchanged)
    serialize(h, base_model, Path(__file__).read_text(encoding='utf-8'))

    if h.distributed:
        import torch.distributed as dist
        dist.barrier()

    # Step 5: Deserialize and evaluate quantized model (unchanged)
    eval_model = deserialize(h, device)
    if h.num_loops > 0:
        eval_model.looping_active = True
    compiled_model = torch.compile(eval_model, dynamic=False, fullgraph=True)
    timed_eval('quantized', eval_val, h, device, val_data, compiled_model)

    if h.sliding_window_enabled:
        timed_eval('quantized_sliding_window', eval_val_sliding, h, device, val_data, eval_model)

    if h.ttt_enabled and h.sliding_window_enabled:
        del eval_model, compiled_model
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        ttt_model = deserialize(h, device)
        if h.num_loops > 0:
            ttt_model.looping_active = True
        timed_eval('quantized_ttt', eval_val_ttt, h, device, val_data, ttt_model)
        del ttt_model

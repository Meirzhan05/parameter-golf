#!/bin/bash
# ---------------------------------------------------------------
# Setup script for 12L QAT Int4-MLP submission
# Run from the repo root: bash records/track_10min_16mb/2026-03-25_QAT_Int4MLP_12L/setup.sh
# ---------------------------------------------------------------
set -e

echo "============================================"
echo " 12L QAT Int4-MLP — Environment Setup"
echo "============================================"

# ---------------------------------------------------------------
# 1. Python dependencies
# ---------------------------------------------------------------
echo ""
echo "[1/3] Installing Python dependencies..."
pip install --upgrade pip -q
pip install numpy tqdm torch sentencepiece huggingface-hub -q
echo "  Done."

# ---------------------------------------------------------------
# 2. Flash Attention 3 (Hopper)
# ---------------------------------------------------------------
echo ""
echo "[2/3] Installing Flash Attention 3..."

if python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null; then
    echo "  Already installed — skipping."
else
    pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
    echo "  Installed."
fi

# ---------------------------------------------------------------
# 3. Dataset + Tokenizer (sp1024)
# ---------------------------------------------------------------
echo ""
echo "[3/3] Downloading dataset (sp1024)..."

cd /workspace/parameter-golf  # adjust if your repo is elsewhere
python3 data/cached_challenge_fineweb.py --variant sp1024

echo "  Done."

# ---------------------------------------------------------------
# Verification
# ---------------------------------------------------------------
echo ""
echo "============================================"
echo " Verification"
echo "============================================"

python3 - << 'PYEOF'
import sys, torch, glob, numpy as np

print(f"Python       : {sys.version.split()[0]}")
print(f"PyTorch      : {torch.__version__}")
print(f"CUDA         : {torch.cuda.is_available()}")
print(f"GPUs         : {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}      : {p.name} ({p.total_mem // 1024**3}GB)")

try:
    from flash_attn_interface import flash_attn_func
    print("FlashAttn3   : OK")
except ImportError:
    print("FlashAttn3   : MISSING!")

train = sorted(glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin"))
val   = sorted(glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"))
print(f"Train shards : {len(train)}")
print(f"Val shards   : {len(val)}")
PYEOF

echo ""
echo "============================================"
echo " Setup complete. Run training with:"
echo ""
echo "   torchrun --nproc_per_node=8 records/track_10min_16mb/2026-03-25_QAT_Int4MLP_12L/train_gpt.py"
echo ""
echo "============================================"

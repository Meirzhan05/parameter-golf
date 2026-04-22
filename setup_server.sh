#!/bin/bash
# =============================================================================
# Server Setup Script for Parameter Golf Tier 1
# =============================================================================
# Run this ONCE after deploying a fresh server (8×H100, 50GB+ disk)
#
# Usage: bash setup_server.sh
# =============================================================================

set -e

echo "============================================"
echo "Parameter Golf - Server Setup"
echo "============================================"

# Step 1: Install dependencies
echo ""
echo "[1/5] Installing Python packages..."
pip install brotli sentencepiece huggingface_hub

# Upgrade PyTorch to 2.9.1+cu128 (required for Flash Attention 3)
echo "Upgrading PyTorch to 2.9.1+cu128..."
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# Install Flash Attention 3 (must match PyTorch 2.9.1+cu128)
echo "Installing Flash Attention 3..."
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/ 2>/dev/null || echo "WARNING: flash_attn_3 pre-built wheel failed, trying pip install..."
pip install flash-attn --no-build-isolation 2>/dev/null || true

# Verify flash attention
python3 -c "from flash_attn_interface import flash_attn_func as f; print('flash_attn_3: OK')" || { echo "ERROR: flash_attn_3 not available!"; exit 1; }

# Step 2: Clone repo and checkout branch
echo ""
echo "[2/5] Setting up repository..."
cd /workspace
if [ ! -d "parameter-golf" ]; then
    git clone https://github.com/Meirzhan05/parameter-golf.git
fi
cd parameter-golf
git fetch origin
git checkout research/beat-sota-pipeline
git pull

# Step 3: Download SP8192 dataset
echo ""
echo "[3/5] Downloading SP8192 tokenizer and dataset..."
python3 download_sp8192.py 80

# Step 4: Verify data
echo ""
echo "[4/5] Verifying data..."
python3 -c "
from pathlib import Path
d = Path('data/datasets/fineweb10B_sp8192')
t = Path('data/tokenizers')
train = list(d.glob('fineweb_train_*.bin'))
val = list(d.glob('fineweb_val_*.bin'))
tok = list(t.glob('*8192*'))
print(f'Train shards: {len(train)}')
print(f'Val shards:   {len(val)}')
print(f'Tokenizer:    {[f.name for f in tok]}')
assert len(train) > 0, 'ERROR: No train shards found!'
assert len(val) > 0, 'ERROR: No val shards found!'
assert len(tok) > 0, 'ERROR: No tokenizer found!'
print('All data verified OK')
"

# Step 5: Quick sanity check
echo ""
echo "[5/5] Sanity check..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo ""
echo "============================================"
echo "Setup complete! Run training with:"
echo ""
echo "cd /workspace/parameter-golf"
echo "PREQUANT_TTT_ENABLED=1 PREQUANT_TTT_OPTIMIZER=adamw PREQUANT_TTT_LR=1e-3 PREQUANT_TTT_EPOCHS=5 TTT_ENABLED=1 torchrun --nproc_per_node=8 run_tier1.py"
echo "============================================"

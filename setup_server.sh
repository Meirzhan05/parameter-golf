#!/bin/bash
# =============================================================================
# Server Setup Script for Parameter Golf
# =============================================================================
# Run after deploying a fresh server (8×H100, 50GB+ disk).
#
# Usage:
#   bash setup_server.sh                                    # use defaults
#   BRANCH=feature/foo bash setup_server.sh                 # custom branch
#   SKIP_BRANCH=1 bash setup_server.sh                      # don't change branch
#   NUM_SHARDS=20 bash setup_server.sh                      # download fewer shards
#   SKIP_DEPS=1 bash setup_server.sh                        # skip pip installs
#   SKIP_DATA=1 bash setup_server.sh                        # skip dataset download
#
# Env vars (all optional):
#   REPO_URL       Default: https://github.com/Meirzhan05/parameter-golf.git
#   WORKSPACE      Default: /workspace
#   BRANCH         Default: (current branch, no checkout)
#   NUM_SHARDS     Default: 80
#   TORCH_VERSION  Default: 2.9.1
#   CUDA_VERSION   Default: cu128
#   FA3_WHEELS     Default: https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
#   SKIP_DEPS      Skip pip installs if set
#   SKIP_BRANCH    Skip git checkout if set
#   SKIP_DATA      Skip data download if set
# =============================================================================

set -e

REPO_URL="${REPO_URL:-https://github.com/Meirzhan05/parameter-golf.git}"
WORKSPACE="${WORKSPACE:-/workspace}"
NUM_SHARDS="${NUM_SHARDS:-80}"
TORCH_VERSION="${TORCH_VERSION:-2.9.1}"
CUDA_VERSION="${CUDA_VERSION:-cu128}"
FA3_WHEELS="${FA3_WHEELS:-https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/}"

echo "============================================"
echo "Parameter Golf - Server Setup"
echo "============================================"
echo "Repo:       $REPO_URL"
echo "Workspace:  $WORKSPACE"
echo "Branch:     ${BRANCH:-<keep current>}"
echo "Num shards: $NUM_SHARDS"
echo "Torch:      $TORCH_VERSION+$CUDA_VERSION"
echo "============================================"

# -----------------------------------------------------------------------------
# Step 1: Install dependencies (skippable)
# -----------------------------------------------------------------------------
if [ -z "$SKIP_DEPS" ]; then
    echo ""
    echo "[1/5] Installing Python packages..."
    pip install brotli sentencepiece huggingface_hub

    # Check if torch is already correct version before reinstalling
    CURRENT_TORCH=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
    if [[ "$CURRENT_TORCH" == "$TORCH_VERSION+$CUDA_VERSION" ]]; then
        echo "  PyTorch $CURRENT_TORCH already installed - skipping."
    else
        echo "  Installing PyTorch $TORCH_VERSION+$CUDA_VERSION (current: $CURRENT_TORCH)..."
        pip install "torch==$TORCH_VERSION" --index-url "https://download.pytorch.org/whl/$CUDA_VERSION"
    fi

    # Install Flash Attention 3 if not already present
    if python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null; then
        echo "  Flash Attention 3 already installed - skipping."
    else
        echo "  Installing Flash Attention 3..."
        pip install flash_attn_3 --no-deps --find-links "$FA3_WHEELS"
    fi

    python3 -c "from flash_attn_interface import flash_attn_func; print('flash_attn_3: OK')" \
        || { echo "ERROR: flash_attn_3 not available!"; exit 1; }
else
    echo "[1/5] Skipping dependencies (SKIP_DEPS set)."
fi

# -----------------------------------------------------------------------------
# Step 2: Clone or update repo (skippable branch checkout)
# -----------------------------------------------------------------------------
echo ""
echo "[2/5] Setting up repository..."
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

REPO_NAME=$(basename "$REPO_URL" .git)
if [ ! -d "$REPO_NAME" ]; then
    git clone "$REPO_URL"
fi
cd "$REPO_NAME"

if [ -z "$SKIP_BRANCH" ] && [ -n "$BRANCH" ]; then
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
elif [ -z "$SKIP_BRANCH" ]; then
    git pull
else
    echo "  SKIP_BRANCH set - keeping current branch: $(git rev-parse --abbrev-ref HEAD)"
fi

# -----------------------------------------------------------------------------
# Step 3: Download dataset (skippable)
# -----------------------------------------------------------------------------
if [ -z "$SKIP_DATA" ]; then
    echo ""
    echo "[3/5] Downloading SP8192 tokenizer and dataset ($NUM_SHARDS shards)..."
    if [ -f "download_sp8192.py" ]; then
        python3 download_sp8192.py "$NUM_SHARDS"
    else
        echo "  download_sp8192.py not found - skipping (you may need to download manually)"
    fi
else
    echo "[3/5] Skipping data download (SKIP_DATA set)."
fi

# -----------------------------------------------------------------------------
# Step 4: Verify data (only if data was supposed to be there)
# -----------------------------------------------------------------------------
echo ""
echo "[4/5] Verifying data..."
python3 - <<'PYEOF'
from pathlib import Path
d = Path('data/datasets/fineweb10B_sp8192')
t = Path('data/tokenizers')
train = sorted(d.glob('fineweb_train_*.bin')) if d.exists() else []
val   = sorted(d.glob('fineweb_val_*.bin'))   if d.exists() else []
tok   = sorted(t.glob('*8192*'))              if t.exists() else []
print(f'Train shards: {len(train)}')
print(f'Val shards:   {len(val)}')
print(f'Tokenizer:    {[f.name for f in tok]}')
if not (train and val and tok):
    print('WARN: data not fully verified (run with SKIP_DATA=0 to download)')
else:
    print('All data verified OK')
PYEOF

# -----------------------------------------------------------------------------
# Step 5: Sanity check GPU + PyTorch
# -----------------------------------------------------------------------------
echo ""
echo "[5/5] Sanity check..."
python3 - <<'PYEOF'
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
PYEOF

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "cd $WORKSPACE/$REPO_NAME"
echo "TTT_ENABLED=1 torchrun --nproc_per_node=8 train_gpt_improved_compressed.py"
echo "============================================"

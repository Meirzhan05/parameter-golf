"""Download SP8192 tokenizer and dataset from kevclark/parameter-golf.

Usage:
    python3 download_sp8192.py            # download 20 train shards (default)
    python3 download_sp8192.py 80         # download all 80 train shards
    python3 download_sp8192.py 10         # download 10 train shards
"""
from huggingface_hub import hf_hub_download, HfApi
import os
import sys
import shutil
from pathlib import Path

NUM_TRAIN_SHARDS = int(sys.argv[1]) if len(sys.argv) > 1 else 20

REPO = "kevclark/parameter-golf"
ROOT = Path(__file__).resolve().parent / "data"
DATASETS_DIR = ROOT / "datasets" / "fineweb10B_sp8192"
TOKENIZERS_DIR = ROOT / "tokenizers"

def download(repo_path, local_path):
    if local_path.exists() or local_path.is_symlink():
        print(f"  exists: {local_path}")
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    parts = repo_path.rsplit("/", 1)
    subfolder = parts[0] if len(parts) > 1 else None
    filename = parts[-1]
    cached = Path(hf_hub_download(REPO, filename, subfolder=subfolder, repo_type="dataset")).resolve()
    # Symlink to HF cache instead of copying — saves disk space
    try:
        os.symlink(cached, local_path)
    except OSError:
        try:
            os.link(cached, local_path)
        except OSError:
            shutil.copy2(cached, local_path)
    print(f"  downloaded: {local_path}")

# Find all sp8192 files
print("Finding sp8192 files...")
api = HfApi()
all_files = api.list_repo_files(REPO, repo_type="dataset")
sp8192_files = sorted([f for f in all_files if "sp8192" in f or "8192_bpe" in f])
print(f"Found {len(sp8192_files)} files")

# Download tokenizer
print("\nDownloading tokenizer...")
tok_files = [f for f in all_files if "8192_bpe" in f]
for f in tok_files:
    name = f.split("/")[-1]
    download(f, TOKENIZERS_DIR / name)

# Download val shards
print("\nDownloading val shards...")
val_files = sorted([f for f in sp8192_files if "val" in f])
for f in val_files:
    name = f.split("/")[-1]
    download(f, DATASETS_DIR / name)

# Download train shards (80 shards)
print("\nDownloading train shards...")
train_files = sorted([f for f in sp8192_files if "train" in f])
print(f"Total train shards available: {len(train_files)}, downloading {NUM_TRAIN_SHARDS}")
for f in train_files[:NUM_TRAIN_SHARDS]:
    name = f.split("/")[-1]
    download(f, DATASETS_DIR / name)

print(f"\nDone! Files in {DATASETS_DIR}:")
print(f"  Train: {len(list(DATASETS_DIR.glob('fineweb_train_*.bin')))} shards")
print(f"  Val: {len(list(DATASETS_DIR.glob('fineweb_val_*.bin')))} shards")
print(f"  Tokenizer: {list(TOKENIZERS_DIR.glob('*8192*'))}")

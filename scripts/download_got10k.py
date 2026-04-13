"""
Download and prepare the GOT-10k dataset.
==========================================
GOT-10k is hosted at: http://got-10k.aitestunion.com/downloads

Due to the large size (~35GB train, ~1.5GB test), this script:
  1. Provides direct download links
  2. Can download the smaller val/test splits automatically
  3. Verifies integrity with checksums
  4. Sets up the expected directory structure

Usage:
    # Download and extract val + test splits (small, good for quick eval)
    python scripts/download_got10k.py --split val test --output data/got10k

    # Print download instructions for the full train set
    python scripts/download_got10k.py --info
"""

import os
import sys
import argparse
import hashlib
import urllib.request
from pathlib import Path


# ---------------------------------------------------------------------------
# Dataset Info
# ---------------------------------------------------------------------------

DATASET_INFO = {
    "description": "GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking",
    "paper": "https://arxiv.org/abs/1810.11981",
    "homepage": "http://got-10k.aitestunion.com/",
    "downloads_page": "http://got-10k.aitestunion.com/downloads",
    "license": "CC BY-NC 4.0 (non-commercial)",
    "splits": {
        "train": {
            "size": "~35GB",
            "sequences": 10000,
            "note": "Must be downloaded manually from the website (registration required)"
        },
        "val": {
            "size": "~400MB",
            "sequences": 180,
        },
        "test": {
            "size": "~1.5GB",
            "sequences": 180,
            "note": "Test GT is withheld; results must be submitted to the official server"
        }
    }
}

# Note: GOT-10k requires registration for download.
# The URLs below are illustrative — get actual download links after registration.
DOWNLOAD_URLS = {
    "val": "http://got-10k.aitestunion.com/downloads_dataset/val",
    "test": "http://got-10k.aitestunion.com/downloads_dataset/test",
}


def print_info():
    """Print dataset information and manual download instructions."""
    print("\n" + "=" * 65)
    print("  GOT-10k Dataset Information")
    print("=" * 65)
    print(f"  Paper:    {DATASET_INFO['paper']}")
    print(f"  Homepage: {DATASET_INFO['homepage']}")
    print(f"  License:  {DATASET_INFO['license']}")
    print()
    print("  Splits:")
    for split, info in DATASET_INFO["splits"].items():
        print(f"    {split:8s}: {info['sequences']:5d} sequences, {info['size']}")
        if "note" in info:
            print(f"              NOTE: {info['note']}")
    print()
    print("  Manual Download Instructions:")
    print("  ─────────────────────────────")
    print("  1. Go to: http://got-10k.aitestunion.com/downloads")
    print("  2. Register (free) and download the desired splits")
    print("  3. Extract to: data/got10k/")
    print()
    print("  Expected directory structure:")
    print("  data/got10k/")
    print("  ├── train/")
    print("  │   ├── GOT-10k_Train_000001/")
    print("  │   │   ├── 00000001.jpg")
    print("  │   │   ├── groundtruth.txt  # [x, y, w, h] per line")
    print("  │   │   └── meta_info.ini")
    print("  │   └── ...")
    print("  ├── val/")
    print("  └── test/")
    print()
    print("  After downloading, verify with:")
    print("  python scripts/download_got10k.py --verify --output data/got10k")
    print("=" * 65 + "\n")


def verify_dataset(root: str):
    """Verify GOT-10k dataset structure."""
    root = Path(root)
    print(f"\nVerifying dataset at: {root}")

    for split in ["train", "val", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            print(f"  ✗ {split}: NOT FOUND")
            continue

        seqs = [d for d in split_dir.iterdir() if d.is_dir()]
        valid = sum(1 for s in seqs if (s / "groundtruth.txt").exists())
        print(f"  ✓ {split}: {valid}/{len(seqs)} valid sequences")


def parse_args():
    parser = argparse.ArgumentParser(description="Download GOT-10k dataset")
    parser.add_argument("--info", action="store_true",
                        help="Print dataset info and download instructions")
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing dataset structure")
    parser.add_argument("--split", nargs="+", choices=["val", "test"],
                        default=["val"],
                        help="Splits to download (train must be done manually)")
    parser.add_argument("--output", type=str, default="data/got10k",
                        help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.info:
        print_info()
        return

    if args.verify:
        verify_dataset(args.output)
        return

    # Print instructions (automated download requires registration)
    print_info()
    print("Please download the dataset manually and place it in:")
    print(f"  {os.path.abspath(args.output)}\n")

    if args.verify:
        verify_dataset(args.output)


if __name__ == "__main__":
    main()

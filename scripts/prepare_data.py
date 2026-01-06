#!/usr/bin/env python
"""
Prepare tmQM dataset for CoordNet training.

This script downloads and preprocesses the tmQM dataset.

Usage:
    python scripts/prepare_data.py --out_dir data

Requirements:
    - Git (for cloning tmQM repository)
    - ~500MB disk space for raw data
    - ~200MB for processed cache
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_tmqm(out_dir: str):
    """Download tmQM dataset from GitHub."""
    raw_dir = Path(out_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    tmqm_dir = raw_dir / "tmQM-master"
    
    if tmqm_dir.exists():
        print(f"tmQM already exists at {tmqm_dir}")
        return tmqm_dir
    
    print("Downloading tmQM dataset...")
    print("Source: https://github.com/bbskjelern/tmQM")
    
    try:
        subprocess.run(
            ["git", "clone", "https://github.com/bbskjelern/tmQM.git", str(tmqm_dir)],
            check=True,
            cwd=str(raw_dir.parent)
        )
        print(f"Downloaded tmQM to {tmqm_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading tmQM: {e}")
        print("\nManual download instructions:")
        print("1. Visit https://github.com/bbskjelern/tmQM")
        print("2. Download and extract to data/raw/tmQM-master/")
        sys.exit(1)
    
    return tmqm_dir


def verify_dataset(tmqm_dir: Path):
    """Verify that required files exist."""
    required_files = [
        "tmQM/tmQM_X1.xyz.gz",
        "tmQM/tmQM_X2.xyz.gz", 
        "tmQM/tmQM_X3.xyz.gz",
        "tmQM/tmQM_y.csv",
        "tmQM/tmQM_X.q"
    ]
    
    missing = []
    for f in required_files:
        if not (tmqm_dir / f).exists():
            missing.append(f)
    
    if missing:
        print("Warning: Missing files:")
        for f in missing:
            print(f"  - {f}")
        return False
    
    print("All required files found.")
    return True


def preprocess_dataset(out_dir: str, num_workers: int = 4):
    """Trigger dataset preprocessing."""
    print("\nPreprocessing dataset...")
    print("This may take 10-30 minutes on first run.")
    
    try:
        from coordnet import TmQMDataset
        dataset = TmQMDataset(root=out_dir)
        print(f"\nDataset ready: {len(dataset)} molecules")
        print(f"Processed data cached at: {out_dir}/processed/")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        print("\nYou can also preprocess by running:")
        print("  python -c \"from coordnet import TmQMDataset; TmQMDataset(root='data')\"")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Prepare tmQM dataset for CoordNet")
    parser.add_argument("--out_dir", type=str, default="data",
                        help="Output directory for dataset (default: data)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for preprocessing")
    parser.add_argument("--skip_preprocess", action="store_true",
                        help="Skip preprocessing (download only)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("CoordNet Data Preparation")
    print("=" * 60)
    
    # Create directory structure
    out_path = Path(args.out_dir)
    (out_path / "raw").mkdir(parents=True, exist_ok=True)
    (out_path / "processed").mkdir(parents=True, exist_ok=True)
    
    # Download
    tmqm_dir = download_tmqm(args.out_dir)
    
    # Verify
    if not verify_dataset(tmqm_dir):
        print("\nDataset verification failed. Please check the download.")
        sys.exit(1)
    
    # Preprocess
    if not args.skip_preprocess:
        preprocess_dataset(args.out_dir, args.num_workers)
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print(f"\nDataset location: {out_path.absolute()}")
    print("\nNext steps:")
    print("  python scripts/train.py --config configs/paper.yaml")


if __name__ == "__main__":
    main()


#!/bin/bash
# CoordNet Paper Reproduction Script
# 
# This script reproduces the main results from the paper.
# 
# Requirements:
#   - Python >= 3.10
#   - PyTorch >= 2.0
#   - PyTorch Geometric >= 2.4
#   - CUDA-compatible GPU (recommended: A100/A800)
#
# Usage:
#   bash scripts/reproduce_paper.sh
#
# Expected runtime: ~6-12 hours on A100 GPU

set -e

echo "=============================================="
echo "CoordNet Paper Reproduction"
echo "=============================================="

# Check Python version
python -c "import sys; assert sys.version_info >= (3, 10), 'Python >= 3.10 required'"

# Set reproducibility environment variables
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42

# Step 1: Prepare data
echo ""
echo "[Step 1/3] Preparing dataset..."
python scripts/prepare_data.py --out_dir data

# Step 2: Train model
echo ""
echo "[Step 2/3] Training CoordNet..."
python scripts/train.py --config configs/paper.yaml

# Step 3: Report results
echo ""
echo "[Step 3/3] Training complete!"
echo ""
echo "Results are saved in the 'runs/' directory."
echo ""
echo "Expected performance (on tmQM test set):"
echo "  - Energy MAE: ~6.2 meV/atom"
echo "  - HOMO-LUMO Gap MAE: ~0.27 eV"
echo "  - Dipole Moment MAE: ~0.42 D"
echo ""
echo "=============================================="


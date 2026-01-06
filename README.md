# CoordNet

<p align="center">
  <img src="docs/architecture.png" width="800"/>
</p>

**CoordNet** is an E(3)-equivariant graph neural network for property prediction of transition metal coordination complexes. The model encodes electronic states from 3D geometry and physical constraints, using only atomic numbers, positions, and molecular charge as inputs.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## Key Features

- **E(3) Equivariance**: Predictions are invariant to rotations and translations
- **Geometry-based Encoding**: Decodes electronic states from 3D structure without relying on heuristic chemical labels
- **Dual Embedding**: Combines atomic number and molecular charge embeddings
- **Multi-task Learning**: Simultaneous prediction of energy, HOMO-LUMO gap, dipole moment, and NPA charges

## Model Variants

| Variant | Description | Config |
|---------|-------------|--------|
| **CoordNet** (default) | Geometry-only model using (z, pos, Q) | `configs/paper.yaml` |
| CoordNet-Geo+FP | Ablation with Morgan fingerprint fusion | `configs/ablation_with_fp.yaml` |

> **Note**: The paper results use the default geometry-only model. The fingerprint fusion variant is provided for ablation studies only.

## Installation

### Option 1: Conda (Recommended)

```bash
git clone https://github.com/1701NJU/CoordNet1.0.git
cd CoordNet1.0
conda env create -f environment.yml
conda activate coordnet
```

### Option 2: Pip

```bash
git clone https://github.com/1701NJU/CoordNet1.0.git
cd CoordNet1.0
pip install -r requirements.txt
```

### PyTorch Geometric Installation Notes

PyTorch Geometric requires matching CUDA/PyTorch versions. If installation fails:

```bash
# Find your PyTorch and CUDA versions
python -c "import torch; print(torch.__version__, torch.version.cuda)"

# Install matching PyG wheels (example for PyTorch 2.0 + CUDA 12.1)
pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu121.html
```

See [PyG Installation Guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for details.

### Dependencies

- Python >= 3.10
- PyTorch >= 2.0
- PyTorch Geometric >= 2.4
- RDKit >= 2023.03 (for data preprocessing)

## Dataset

We use the [tmQM dataset](https://github.com/bbskjelern/tmQM) containing ~86,000 DFT-optimized transition metal complexes.

### Automatic Setup

```bash
python scripts/prepare_data.py --out_dir data
```

### Manual Setup

```bash
mkdir -p data/raw
cd data/raw
git clone https://github.com/bbskjelern/tmQM.git tmQM-master
```

The dataset will be automatically preprocessed on first training run. Preprocessing takes 10-30 minutes and creates a cached file at `data/processed/tmqm_data.pt`.

## Training

### Paper Reproduction

```bash
# Full reproduction pipeline
bash scripts/reproduce_paper.sh

# Or step by step:
python scripts/prepare_data.py --out_dir data
python scripts/train.py --config configs/paper.yaml
```

### Custom Training

```bash
python scripts/train.py --config configs/default.yaml
```

### Configuration

Key hyperparameters in `configs/paper.yaml`:

```yaml
model:
  hidden_dim: 128      # Feature dimension
  num_layers: 3        # Number of PaiNN layers
  cutoff: 5.0          # Neighbor cutoff (Å)
  use_fp: false        # Geometry-only (paper default)

training:
  batch_size: 32
  lr: 5.0e-4
  epochs: 300
  max_grad_norm: 10.0
```

## Model Performance

Results on tmQM test set (random 80/10/10 split, seed=42):

| Property | MAE | Unit | Notes |
|----------|-----|------|-------|
| Electronic Energy | 6.2 | meV/atom | Per-atom normalized |
| HOMO-LUMO Gap | 0.27 | eV | |
| Dipole Moment | 0.42 | D | |

<details>
<summary>Additional metrics (click to expand)</summary>

| Property | MAE (total) | Unit |
|----------|-------------|------|
| Electronic Energy | 0.31 | eV/molecule |
| HOMO-LUMO Gap | 0.010 | Hartree |

</details>

## Usage

### Inference

```python
import torch
from coordnet import CoordNet, AtomRefCalculator

# Load model
atom_ref = AtomRefCalculator().load('runs/best/atom_ref.npz')
model = CoordNet(atom_ref=atom_ref)
model.load_state_dict(torch.load('runs/best/best_model.pt')['model_state_dict'])
model.eval()

# Predict
with torch.no_grad():
    output = model(batch)
    energy = output['energy']      # [B] eV
    gap = output['gap']            # [B] Hartree (multiply by 27.2114 for eV)
    dipole = output['dipole']      # [B] Debye
    npa = output['npa']            # [N] e
```

### Custom Data

CoordNet requires minimal input: atomic numbers, positions, and molecular charge.

```python
from torch_geometric.data import Data

# Minimal input (geometry-only, paper model)
data = Data(
    z=torch.tensor([26, 6, 7, 8]),           # Atomic numbers
    pos=torch.tensor([[0, 0, 0], ...]),      # Coordinates (Å)
    charge=torch.tensor([0]),                 # Molecular charge
    batch=torch.zeros(4, dtype=torch.long)   # Batch indices
)
```

## Architecture

### Model Components

| Component | Description |
|-----------|-------------|
| `DualEmbedding` | Atom (Z) + charge (Q) embedding |
| `RadialBasisFunctions` | Gaussian RBF distance encoding |
| `PaiNNLayer` | E(3)-equivariant message passing |
| `GeometryOnlyReadout` | Graph-level pooling (default) |
| `EnergyHead` / `GapHead` | Scalar property prediction |
| `DipoleHead` | E(3)-equivariant vector prediction |

### E(3) Equivariance

The model maintains E(3) equivariance through:

1. **Scalar features** $\mathbf{s}_i \in \mathbb{R}^F$: Invariant under rotations
2. **Vector features** $\mathbf{v}_i \in \mathbb{R}^{3 \times F}$: Transform as SO(3) vectors

### Message Passing

For each edge $(i, j)$:

$$\Delta \mathbf{s}_i = \sum_{j} \phi(\mathbf{s}_j) \odot W(d_{ij})$$

$$\Delta \mathbf{v}_i = \sum_{j} \left[\mathbf{m}^{sv}_{ij} + (\mathbf{v}_j \cdot \hat{r}_{ij}) \mathbf{m}^{vv}_{ij}\right] \otimes \hat{r}_{ij}$$

### Atom Reference Correction

Energy prediction uses per-element reference energies fitted on training data:

$$E_{\text{pred}} = E_{\text{NN}} + \sum_i E_{\text{ref}}(Z_i)$$

## Citation

If you use this code in your research, please cite:

```bibtex
@article{coordnet2026,
  title={CoordNet: E(3)-Equivariant Neural Network for Transition Metal Complex Property Prediction},
  author={Luo, Wenlin},
  journal={},
  year={2026},
  note={Code available at https://github.com/1701NJU/CoordNet1.0}
}
```

## Contact

- **Author**: Wenlin Luo
- **Email**: luowenlin862@gmail.com
- **Issues**: [GitHub Issues](https://github.com/1701NJU/CoordNet1.0/issues)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [PaiNN](https://github.com/atomistic-machine-learning/schnetpack) - Base equivariant architecture
- [tmQM](https://github.com/bbskjelern/tmQM) - Transition metal complex dataset
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - Graph neural network framework

# CoordNet

**CoordNet** (Coordination Chemistry Neural Network) is an E(3)-equivariant graph neural network designed for property prediction of transition metal coordination complexes.

## Architecture

<p align="center">
  <img src="docs/architecture.png" width="800"/>
</p>

### Key Features

- **E(3) Equivariance**: Predictions are invariant to rotations and translations
- **Dual Embedding**: Combines atom type and molecular charge embeddings
- **Late Fusion**: Integrates 3D geometric features with Morgan fingerprints
- **Multi-task Learning**: Simultaneous prediction of energy, HOMO-LUMO gap, dipole moment, and NPA charges

### Model Components

| Component | Description |
|-----------|-------------|
| `DualEmbedding` | Atom + charge embedding layer |
| `RadialBasisFunctions` | Gaussian RBF distance encoding |
| `PaiNNLayer` | E(3)-equivariant message passing |
| `LateFusionReadout` | Geometry-fingerprint fusion |
| `EnergyHead` / `GapHead` | Scalar property prediction |
| `DipoleHead` | E(3)-equivariant vector prediction |

## Installation

```bash
git clone https://github.com/luowenlin862/CoordNet.git
cd CoordNet
pip install -r requirements.txt
```

### Dependencies

- Python >= 3.10
- PyTorch >= 2.0
- PyTorch Geometric >= 2.4
- RDKit >= 2023.03

## Dataset

We use the [tmQM dataset](https://github.com/bbskjelern/tmQM) containing ~86,000 DFT-optimized transition metal complexes.

### Setup

1. Download tmQM:
```bash
cd data/raw
git clone https://github.com/bbskjelern/tmQM.git tmQM-master
```

2. The dataset will be automatically processed on first run.

## Training

```bash
python scripts/train.py --config configs/default.yaml
```

### Configuration

Key hyperparameters in `configs/default.yaml`:

```yaml
model:
  hidden_dim: 128      # Feature dimension
  num_layers: 3        # Number of PaiNN layers
  cutoff: 5.0          # Neighbor cutoff (Å)

training:
  batch_size: 32
  lr: 5.0e-4
  epochs: 300
  max_grad_norm: 10.0  # Gradient clipping

loss_weights:
  energy: 1.0          # Primary task
  gap: 0.1             # Auxiliary tasks
  dipole: 0.1
  npa: 0.05
```

## Model Performance

| Property | MAE | Unit |
|----------|-----|------|
| Electronic Energy | 0.31 | eV |
| HOMO-LUMO Gap | 0.44 | eV |
| Dipole Moment | 0.65 | D |

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
    gap = output['gap']            # [B] Hartree
    dipole = output['dipole']      # [B] Debye
    npa = output['npa']            # [N] e
```

### Custom Data

```python
from torch_geometric.data import Data

data = Data(
    z=torch.tensor([26, 6, 7, ...]),           # Atomic numbers
    pos=torch.tensor([[0, 0, 0], ...]),        # Coordinates (Å)
    charge=torch.tensor([0]),                   # Molecular charge
    mol_fp=torch.zeros(1024),                   # Morgan fingerprint
)
```

## Mathematical Foundation

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

$$E_{pred} = E_{NN} + \sum_i E_{ref}(Z_i)$$

## Citation

If you use this code in your research, please cite:

```bibtex
@article{coordnet2026,
  title={CoordNet: E(3)-Equivariant Neural Network for Transition Metal Complex Property Prediction},
  author={Luo, Wenlin},
  journal={},
  year={2026}
}
```

## Contact

- **Author**: Wenlin Luo
- **Email**: luowenlin862@gmail.com

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [PaiNN](https://github.com/atomistic-machine-learning/schnetpack) - Base architecture
- [tmQM](https://github.com/bbskjelern/tmQM) - Dataset
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - Graph neural network framework

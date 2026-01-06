# Data Directory

This directory contains the tmQM dataset for CoordNet training.

## Directory Structure

```
data/
├── raw/
│   └── tmQM-master/
│       └── tmQM/
│           ├── tmQM_X1.xyz.gz    # Molecular structures (part 1)
│           ├── tmQM_X2.xyz.gz    # Molecular structures (part 2)
│           ├── tmQM_X3.xyz.gz    # Molecular structures (part 3)
│           ├── tmQM_y.csv        # Target properties
│           └── tmQM_X.q          # NPA charges
├── processed/
│   └── tmqm_data.pt              # Preprocessed PyG dataset (auto-generated)
└── README.md
```

## Setup Instructions

### Option 1: Automatic Download (Recommended)

```bash
python scripts/prepare_data.py --out_dir data
```

### Option 2: Manual Download

1. Clone the tmQM repository:
```bash
cd data/raw
git clone https://github.com/bbskjelern/tmQM.git tmQM-master
```

2. The dataset will be automatically preprocessed on first training run.

## Dataset Information

- **Source**: [tmQM Dataset](https://github.com/bbskjelern/tmQM)
- **Size**: ~86,000 DFT-optimized transition metal complexes
- **Properties**:
  - Electronic energy (Hartree → converted to eV)
  - HOMO-LUMO gap (Hartree)
  - Dipole moment (Debye)
  - NPA charges (e)

## References

```bibtex
@article{tmQM2020,
  title={The tmQM dataset — quantum geometries and properties of 86k transition metal complexes},
  author={Balcells, David and Skjelstad, Bastian Bjerkem},
  journal={Journal of Chemical Information and Modeling},
  volume={60},
  pages={6135--6146},
  year={2020}
}
```


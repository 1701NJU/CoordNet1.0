import os
import sys
import argparse
import torch
from torch_geometric.data import Data
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coordnet import CoordNet, AtomRefCalculator


ELEMENT_TO_Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53,
                'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
                'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48,
                'La': 57, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80}


def load_xyz(filepath):
    elements, coords = [], []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        n_atoms = int(lines[0].strip())
        for line in lines[2:2+n_atoms]:
            parts = line.strip().split()
            elements.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])
    z = [ELEMENT_TO_Z.get(e, ELEMENT_TO_Z.get(e.capitalize(), 0)) for e in elements]
    return np.array(z), np.array(coords, dtype=np.float32)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    atom_ref = AtomRefCalculator().load(args.atom_ref)
    model = CoordNet(atom_ref=atom_ref).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    z, pos = load_xyz(args.xyz)
    
    data = Data(
        z=torch.tensor(z, dtype=torch.long),
        pos=torch.tensor(pos, dtype=torch.float32),
        charge=torch.tensor([args.charge], dtype=torch.long),
        mol_fp=torch.zeros(1024, dtype=torch.float32),
        batch=torch.zeros(len(z), dtype=torch.long)
    )
    data = data.to(device)
    
    with torch.no_grad():
        output = model(data)
    
    print(f"\nPredictions for {args.xyz}:")
    print(f"  Energy: {output['energy'].item():.2f} eV")
    print(f"  HOMO-LUMO Gap: {output['gap'].item() * 27.2114:.3f} eV ({output['gap'].item():.4f} Ha)")
    print(f"  Dipole Moment: {output['dipole'].item():.3f} D")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xyz', type=str, required=True, help='Path to XYZ file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--atom_ref', type=str, required=True, help='Path to atom_ref.npz')
    parser.add_argument('--charge', type=int, default=0, help='Molecular charge')
    main(parser.parse_args())


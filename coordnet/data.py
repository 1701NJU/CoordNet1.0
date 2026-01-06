import os
import gzip
import re
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
from tqdm import tqdm

import torch
from torch_geometric.data import Data, InMemoryDataset
from sklearn.linear_model import Ridge

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


HARTREE_TO_EV = 27.2114

ELEMENT_LIST = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm'
]

ELEMENT_TO_Z = {elem: i + 1 for i, elem in enumerate(ELEMENT_LIST)}
MAX_ATOMIC_NUM = 100


class MorganFeaturizer:
    def __init__(self, radius: int = 2, n_bits: int = 1024):
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for Morgan fingerprint generation")
        self.radius = radius
        self.n_bits = n_bits
        self.fp_gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    
    def featurize(self, smiles: Optional[str]) -> tuple:
        if smiles is None or smiles == '' or (isinstance(smiles, float) and np.isnan(smiles)):
            return np.zeros(self.n_bits, dtype=np.float32), False
        
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return np.zeros(self.n_bits, dtype=np.float32), False
            
            try:
                mol.UpdatePropertyCache(strict=False)
                Chem.FastFindRings(mol)
            except Exception:
                pass
            
            try:
                fp = self.fp_gen.GetFingerprint(mol)
                fp_array = np.zeros(self.n_bits, dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, fp_array)
                return fp_array, True
            except Exception:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
                fp_array = np.zeros(self.n_bits, dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, fp_array)
                return fp_array, True
        except Exception:
            return np.zeros(self.n_bits, dtype=np.float32), False


class AtomRefCalculator:
    def __init__(self, max_atomic_num: int = MAX_ATOMIC_NUM):
        self.max_atomic_num = max_atomic_num
        self.atom_ref = None
        self.fitted = False
    
    def fit(self, atomic_numbers_list: list, charges: np.ndarray, energies: np.ndarray):
        n_samples = len(atomic_numbers_list)
        X = np.zeros((n_samples, self.max_atomic_num), dtype=np.float64)
        y = np.asarray(energies, dtype=np.float64)
        
        for i, z_array in enumerate(atomic_numbers_list):
            for z in z_array:
                if 1 <= z <= self.max_atomic_num:
                    X[i, z - 1] += 1
        
        reg = Ridge(alpha=1e-6, fit_intercept=False)
        reg.fit(X, y)
        self.atom_ref = reg.coef_
        self.fitted = True
        
        y_pred = reg.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        residuals = y - y_pred
        
        print(f"AtomRef fit: R²={r2:.6f}, residual std={residuals.std():.2f} eV")
        return self
    
    def get_reference_energy(self, atomic_numbers: np.ndarray, charge: float = 0) -> float:
        if not self.fitted:
            raise RuntimeError("AtomRefCalculator not fitted")
        energy = 0.0
        for z in atomic_numbers:
            if 1 <= z <= self.max_atomic_num:
                energy += self.atom_ref[z - 1]
        return energy
    
    def get_reference_energy_batch(self, atomic_numbers: torch.Tensor, batch: torch.Tensor, charges: torch.Tensor = None) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("AtomRefCalculator not fitted")
        
        device = atomic_numbers.device
        batch_size = batch.max().item() + 1
        atom_ref_tensor = torch.tensor(self.atom_ref, dtype=torch.float32, device=device)
        valid_z = torch.clamp(atomic_numbers - 1, 0, self.max_atomic_num - 1)
        ref_per_atom = atom_ref_tensor[valid_z]
        ref_per_mol = torch.zeros(batch_size, dtype=torch.float32, device=device)
        ref_per_mol.scatter_add_(0, batch, ref_per_atom)
        return ref_per_mol
    
    def save(self, path: str):
        if not self.fitted:
            raise RuntimeError("AtomRefCalculator not fitted")
        np.savez(path, atom_ref=self.atom_ref)
    
    def load(self, path: str):
        data = np.load(path)
        self.atom_ref = data['atom_ref']
        self.fitted = True
        return self


class TmQMDataset(InMemoryDataset):
    def __init__(self, root: str, transform=None, pre_transform=None, pre_filter=None, fp_radius: int = 2, fp_bits: int = 1024):
        self.fp_radius = fp_radius
        self.fp_bits = fp_bits
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
    
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw', 'tmQM-master', 'tmQM')
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')
    
    @property
    def raw_file_names(self) -> List[str]:
        return ['tmQM_X1.xyz.gz', 'tmQM_X2.xyz.gz', 'tmQM_X3.xyz.gz', 'tmQM_y.csv', 'tmQM_X.q']
    
    @property
    def processed_file_names(self) -> List[str]:
        return ['tmqm_data.pt']
    
    def download(self):
        raise RuntimeError("tmQM data not found. Download from https://github.com/bbskjelern/tmQM")
    
    def process(self):
        print("Processing tmQM dataset...")
        featurizer = MorganFeaturizer(radius=self.fp_radius, n_bits=self.fp_bits)
        
        csv_path = os.path.join(self.raw_dir, 'tmQM_y.csv')
        df = pd.read_csv(csv_path, sep=';')
        csv_data = {row['CSD_code']: row for _, row in df.iterrows()}
        
        q_path = os.path.join(self.raw_dir, 'tmQM_X.q')
        npa_data = self._parse_npa_file(q_path)
        
        xyz_files = ['tmQM_X1.xyz.gz', 'tmQM_X2.xyz.gz', 'tmQM_X3.xyz.gz']
        data_list = []
        
        for xyz_file in xyz_files:
            xyz_path = os.path.join(self.raw_dir, xyz_file)
            molecules = self._parse_xyz_file(xyz_path)
            
            for mol in tqdm(molecules, desc=f"Processing {xyz_file}"):
                csd_code = mol['csd_code']
                if csd_code not in csv_data:
                    continue
                
                csv_row = csv_data[csd_code]
                npa_charges = npa_data.get(csd_code, None)
                if npa_charges is not None and len(npa_charges) != len(mol['z']):
                    npa_charges = None
                
                smiles = csv_row.get('SMILES', '')
                fp, fp_valid = featurizer.featurize(smiles)
                mnd_class = max(0, min(mol['mnd'] - 1, 19))
                
                data = Data(
                    z=torch.tensor(mol['z'], dtype=torch.long),
                    pos=torch.tensor(mol['pos'], dtype=torch.float32),
                    charge=torch.tensor([mol['charge']], dtype=torch.long),
                    mol_fp=torch.tensor(fp, dtype=torch.float32),
                    fp_valid=torch.tensor([fp_valid], dtype=torch.bool),
                    y_energy=torch.tensor([csv_row['Electronic_E'] * HARTREE_TO_EV], dtype=torch.float32),
                    y_gap=torch.tensor([csv_row['HL_Gap']], dtype=torch.float32),
                    y_dipole=torch.tensor([csv_row['Dipole_M']], dtype=torch.float32),
                    y_mnd=torch.tensor([mnd_class], dtype=torch.long),
                    y_npa=torch.tensor(npa_charges, dtype=torch.float32) if npa_charges is not None else None,
                    csd_code=csd_code
                )
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)
        
        print(f"Processed {len(data_list)} molecules")
        self.save(data_list, self.processed_paths[0])
    
    def _parse_xyz_file(self, filepath: str) -> List[Dict]:
        molecules = []
        open_fn = gzip.open if filepath.endswith('.gz') else open
        
        with open_fn(filepath, 'rt') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                try:
                    n_atoms = int(line.strip())
                except ValueError:
                    continue
                
                meta_line = f.readline().strip()
                meta = self._parse_meta_line(meta_line)
                
                z_list, pos_list = [], []
                for _ in range(n_atoms):
                    atom_line = f.readline().strip().split()
                    element = atom_line[0]
                    coords = [float(x) for x in atom_line[1:4]]
                    z = ELEMENT_TO_Z.get(element, ELEMENT_TO_Z.get(element.capitalize(), 0))
                    z_list.append(z)
                    pos_list.append(coords)
                
                molecules.append({
                    'csd_code': meta.get('CSD_code', ''),
                    'charge': meta.get('q', 0),
                    'mnd': meta.get('MND', 4),
                    'z': np.array(z_list, dtype=np.int64),
                    'pos': np.array(pos_list, dtype=np.float32)
                })
        return molecules
    
    def _parse_meta_line(self, line: str) -> Dict:
        meta = {}
        match = re.search(r'CSD_code\s*=\s*(\w+)', line)
        if match:
            meta['CSD_code'] = match.group(1)
        match = re.search(r'\bq\s*=\s*(-?\d+)', line)
        if match:
            meta['q'] = int(match.group(1))
        match = re.search(r'MND\s*=\s*(\d+)', line)
        if match:
            meta['MND'] = int(match.group(1))
        return meta
    
    def _parse_npa_file(self, filepath: str) -> Dict[str, List[float]]:
        npa_data = {}
        current_code, current_charges = None, []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('CSD_code'):
                    if current_code is not None and current_charges:
                        npa_data[current_code] = current_charges
                    match = re.search(r'CSD_code\s*=\s*(\w+)', line)
                    if match:
                        current_code = match.group(1)
                        current_charges = []
                elif line and current_code is not None:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            current_charges.append(float(parts[1]))
                        except ValueError:
                            pass
        
        if current_code is not None and current_charges:
            npa_data[current_code] = current_charges
        return npa_data
    
    def get_split_indices(self, split_type: str = 'random', train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(self)
        np.random.seed(seed)
        indices = np.random.permutation(n)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        return indices[:train_end], indices[train_end:val_end], indices[val_end:]


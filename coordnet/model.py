"""
CoordNet: E(3)-Equivariant Neural Network for Transition Metal Complex Property Prediction

Copyright (c) 2026 Wenlin Luo (luowenlin862@gmail.com)
Licensed under the MIT License.
"""

import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.nn import radius_graph


class RadialBasisFunctions(nn.Module):
    """Gaussian radial basis functions for distance encoding."""
    
    def __init__(self, num_rbf: int = 20, cutoff: float = 5.0):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        offset = torch.linspace(0, cutoff, num_rbf)
        self.register_buffer('offset', offset)
        self.width = (offset[1] - offset[0]).item()
    
    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.unsqueeze(-1)
        return torch.exp(-0.5 * ((dist - self.offset) / self.width) ** 2)


class CosineCutoff(nn.Module):
    """Smooth cosine cutoff function."""
    
    def __init__(self, cutoff: float = 5.0):
        super().__init__()
        self.cutoff = cutoff
    
    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        cutoff = 0.5 * (1 + torch.cos(torch.pi * dist / self.cutoff))
        return cutoff * (dist <= self.cutoff).float()


class DualEmbedding(nn.Module):
    """Dual embedding layer combining atomic number and molecular charge."""
    
    def __init__(self, num_elements: int = 100, num_charges: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.atom_emb = nn.Embedding(num_elements, hidden_dim)
        self.charge_emb = nn.Embedding(num_charges, hidden_dim)
        nn.init.xavier_uniform_(self.atom_emb.weight)
        nn.init.xavier_uniform_(self.charge_emb.weight)
    
    def forward(self, z: torch.Tensor, charge: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h_atom = self.atom_emb(z - 1)
        h_charge = self.charge_emb(charge)[batch]
        return h_atom + h_charge


class PaiNNMessage(nn.Module):
    """E(3)-equivariant message passing."""
    
    def __init__(self, hidden_dim: int, num_rbf: int = 20):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scalar_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 3)
        )
        self.rbf_proj = nn.Linear(num_rbf, hidden_dim * 3)
    
    def forward(self, s, v, edge_index, edge_rbf, edge_cutoff, edge_vec):
        i, j = edge_index
        W = self.rbf_proj(edge_rbf) * edge_cutoff.unsqueeze(-1)
        x = self.scalar_mlp(s[j]) * W
        x_ss, x_sv, x_vv = x.chunk(3, dim=-1)
        
        ds = scatter(x_ss, i, dim=0, dim_size=s.size(0), reduce='add')
        
        dv1 = x_sv.unsqueeze(1) * edge_vec.unsqueeze(-1)
        dv1 = scatter(dv1, i, dim=0, dim_size=s.size(0), reduce='add')
        
        v_j = v[j]
        inner = (v_j * edge_vec.unsqueeze(-1)).sum(dim=1)
        dv2 = (inner * x_vv).unsqueeze(1) * edge_vec.unsqueeze(-1)
        dv2 = scatter(dv2, i, dim=0, dim_size=s.size(0), reduce='add')
        
        return ds, dv1 + dv2


class PaiNNUpdate(nn.Module):
    """E(3)-equivariant update block."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vec_proj = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 3)
        )
    
    def forward(self, s: torch.Tensor, v: torch.Tensor):
        v_proj = self.vec_proj(v)
        v_u, v_v = v_proj.chunk(2, dim=-1)
        v_v_norm = torch.norm(v_v, dim=1)
        
        s_input = torch.cat([s, v_v_norm], dim=-1)
        s_out = self.scalar_mlp(s_input)
        a_ss, a_sv, a_vv = s_out.chunk(3, dim=-1)
        
        return s + a_ss, v + a_vv.unsqueeze(1) * v_u + a_sv.unsqueeze(1) * v_v


class PaiNNLayer(nn.Module):
    """Single PaiNN interaction layer."""
    
    def __init__(self, hidden_dim: int, num_rbf: int = 20):
        super().__init__()
        self.message = PaiNNMessage(hidden_dim, num_rbf)
        self.update = PaiNNUpdate(hidden_dim)
    
    def forward(self, s, v, edge_index, edge_rbf, edge_cutoff, edge_vec):
        ds, dv = self.message(s, v, edge_index, edge_rbf, edge_cutoff, edge_vec)
        s, v = s + ds, v + dv
        return self.update(s, v)


class GeometryOnlyReadout(nn.Module):
    """Geometry-only readout (no fingerprint fusion)."""
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, geom_vec: torch.Tensor, mol_fp: torch.Tensor = None) -> torch.Tensor:
        return self.mlp(geom_vec)


class LateFusionReadout(nn.Module):
    """Late fusion readout combining geometry and Morgan fingerprints.
    
    Note: This is an optional ablation/engineering variant. 
    The default CoordNet uses geometry-only readout.
    """
    
    def __init__(self, geom_dim: int = 128, fp_dim: int = 1024, hidden_dim: int = 128):
        super().__init__()
        self.fp_proj = nn.Linear(fp_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(geom_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, geom_vec: torch.Tensor, mol_fp: torch.Tensor) -> torch.Tensor:
        fp_compressed = self.fp_proj(mol_fp)
        return self.mlp(torch.cat([geom_vec, fp_compressed], dim=-1))


class EnergyHead(nn.Module):
    """Energy prediction head."""
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)


class GapHead(nn.Module):
    """HOMO-LUMO gap prediction head (output in Hartree)."""
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)


class DipoleHead(nn.Module):
    """E(3)-equivariant dipole moment prediction head."""
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.lin = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, v: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        v_weighted = self.lin(v).squeeze(-1)
        batch_size = batch.max().item() + 1
        dipole_vec = scatter(v_weighted, batch, dim=0, dim_size=batch_size, reduce='add')
        return torch.norm(dipole_vec, dim=-1)


class NPAHead(nn.Module):
    """NPA charge prediction head."""
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.mlp(s).squeeze(-1)


def charge_to_index(charge: torch.Tensor, max_abs_charge: int = 2) -> torch.Tensor:
    """Convert molecular charge to embedding index."""
    if charge.dim() == 2:
        charge = charge.squeeze(-1)
    charge_clamped = torch.clamp(charge, -max_abs_charge, max_abs_charge)
    return (charge_clamped + max_abs_charge).long()


class CoordNet(nn.Module):
    """CoordNet: E(3)-Equivariant Neural Network for Transition Metal Complexes.
    
    This model predicts properties of transition metal coordination complexes
    using only 3D geometry (atomic numbers, positions) and molecular charge.
    
    Args:
        hidden_dim: Hidden feature dimension (default: 128)
        num_layers: Number of PaiNN interaction layers (default: 3)
        num_rbf: Number of radial basis functions (default: 20)
        cutoff: Neighbor cutoff distance in Angstroms (default: 5.0)
        num_elements: Maximum atomic number (default: 100)
        num_charges: Number of charge states supported (default: 5, for -2 to +2)
        use_fp: Whether to use Morgan fingerprint fusion (default: False)
                Set to True for CoordNet-Geo+FP variant (ablation study)
        fp_dim: Morgan fingerprint dimension (default: 1024, only used if use_fp=True)
        atom_ref: AtomRefCalculator for energy baseline correction
    
    Model Variants:
        - CoordNet (default, use_fp=False): Pure geometry-based model
          Uses only atomic numbers (z), positions (pos), and molecular charge (Q)
          
        - CoordNet-Geo+FP (use_fp=True): With Morgan fingerprint fusion
          Ablation variant that additionally uses 2D topological fingerprints
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_rbf: int = 20,
        cutoff: float = 5.0,
        num_elements: int = 100,
        num_charges: int = 5,
        use_fp: bool = False,
        fp_dim: int = 1024,
        atom_ref=None
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cutoff = cutoff
        self.use_fp = use_fp
        self.atom_ref = atom_ref
        
        # Core E(3)-equivariant components
        self.embedding = DualEmbedding(num_elements, num_charges, hidden_dim)
        self.rbf = RadialBasisFunctions(num_rbf, cutoff)
        self.cutoff_fn = CosineCutoff(cutoff)
        self.layers = nn.ModuleList([PaiNNLayer(hidden_dim, num_rbf) for _ in range(num_layers)])
        
        # Readout: geometry-only (default) or with fingerprint fusion (ablation)
        if use_fp:
            self.readout = LateFusionReadout(hidden_dim, fp_dim, hidden_dim)
        else:
            self.readout = GeometryOnlyReadout(hidden_dim)
        
        # Prediction heads
        self.energy_head = EnergyHead(hidden_dim)
        self.gap_head = GapHead(hidden_dim)
        self.dipole_head = DipoleHead(hidden_dim)
        self.npa_head = NPAHead(hidden_dim)
    
    def forward(self, batch):
        """Forward pass.
        
        Args:
            batch: PyTorch Geometric batch with:
                - z: Atomic numbers [N]
                - pos: Atomic positions [N, 3] in Angstroms
                - charge: Molecular charge [B]
                - batch: Batch indices [N]
                - mol_fp (optional): Morgan fingerprints [B, fp_dim] (only if use_fp=True)
        
        Returns:
            dict with:
                - energy: Electronic energy [B] in eV
                - gap: HOMO-LUMO gap [B] in Hartree
                - dipole: Dipole moment [B] in Debye
                - npa: NPA charges [N] in e
        """
        z = batch.z
        pos = batch.pos
        charge = batch.charge
        batch_idx = batch.batch
        
        batch_size = charge.shape[0]
        
        # Get fingerprints if using FP fusion
        mol_fp = None
        if self.use_fp:
            mol_fp = batch.mol_fp
            if mol_fp.dim() == 1:
                fp_dim = mol_fp.shape[0] // batch_size
                mol_fp = mol_fp.view(batch_size, fp_dim)
        
        # Embedding
        charge_idx = charge_to_index(charge)
        s = self.embedding(z, charge_idx, batch_idx)
        v = torch.zeros(s.size(0), 3, self.hidden_dim, device=s.device)
        
        # Build graph
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch_idx)
        i, j = edge_index
        edge_vec = pos[j] - pos[i]
        edge_dist = torch.norm(edge_vec, dim=1)
        edge_vec = edge_vec / (edge_dist.unsqueeze(-1) + 1e-8)
        
        # Distance encoding
        edge_rbf = self.rbf(edge_dist)
        edge_cutoff = self.cutoff_fn(edge_dist)
        
        # Message passing
        for layer in self.layers:
            s, v = layer(s, v, edge_index, edge_rbf, edge_cutoff, edge_vec)
        
        # Readout
        s_mol = scatter(s, batch_idx, dim=0, reduce='add')
        fused = self.readout(s_mol, mol_fp)
        
        # Predictions
        energy = self.energy_head(fused)
        gap = self.gap_head(fused)
        dipole = self.dipole_head(v, batch_idx)
        npa = self.npa_head(s)
        
        # Atom reference correction
        if self.atom_ref is not None:
            ref_energy = self.atom_ref.get_reference_energy_batch(z, batch_idx)
            energy = energy + ref_energy
        
        return {'energy': energy, 'gap': gap, 'dipole': dipole, 'npa': npa}

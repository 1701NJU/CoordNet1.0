import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.nn import radius_graph


class RadialBasisFunctions(nn.Module):
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
    def __init__(self, cutoff: float = 5.0):
        super().__init__()
        self.cutoff = cutoff
    
    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        cutoff = 0.5 * (1 + torch.cos(torch.pi * dist / self.cutoff))
        return cutoff * (dist <= self.cutoff).float()


class DualEmbedding(nn.Module):
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
    def __init__(self, hidden_dim: int, num_rbf: int = 20):
        super().__init__()
        self.message = PaiNNMessage(hidden_dim, num_rbf)
        self.update = PaiNNUpdate(hidden_dim)
    
    def forward(self, s, v, edge_index, edge_rbf, edge_cutoff, edge_vec):
        ds, dv = self.message(s, v, edge_index, edge_rbf, edge_cutoff, edge_vec)
        s, v = s + ds, v + dv
        return self.update(s, v)


class LateFusionReadout(nn.Module):
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
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.lin = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, v: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        v_weighted = self.lin(v).squeeze(-1)
        batch_size = batch.max().item() + 1
        dipole_vec = scatter(v_weighted, batch, dim=0, dim_size=batch_size, reduce='add')
        return torch.norm(dipole_vec, dim=-1)


class NPAHead(nn.Module):
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
    if charge.dim() == 2:
        charge = charge.squeeze(-1)
    charge_clamped = torch.clamp(charge, -max_abs_charge, max_abs_charge)
    return (charge_clamped + max_abs_charge).long()


class CoordNet(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_rbf: int = 20,
        cutoff: float = 5.0,
        num_elements: int = 100,
        num_charges: int = 5,
        fp_dim: int = 1024,
        atom_ref=None
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cutoff = cutoff
        self.atom_ref = atom_ref
        
        self.embedding = DualEmbedding(num_elements, num_charges, hidden_dim)
        self.rbf = RadialBasisFunctions(num_rbf, cutoff)
        self.cutoff_fn = CosineCutoff(cutoff)
        self.layers = nn.ModuleList([PaiNNLayer(hidden_dim, num_rbf) for _ in range(num_layers)])
        self.readout = LateFusionReadout(hidden_dim, fp_dim, hidden_dim)
        self.energy_head = EnergyHead(hidden_dim)
        self.gap_head = GapHead(hidden_dim)
        self.dipole_head = DipoleHead(hidden_dim)
        self.npa_head = NPAHead(hidden_dim)
    
    def forward(self, batch):
        z = batch.z
        pos = batch.pos
        charge = batch.charge
        batch_idx = batch.batch
        
        batch_size = charge.shape[0]
        mol_fp = batch.mol_fp
        if mol_fp.dim() == 1:
            fp_dim = mol_fp.shape[0] // batch_size
            mol_fp = mol_fp.view(batch_size, fp_dim)
        
        charge_idx = charge_to_index(charge)
        s = self.embedding(z, charge_idx, batch_idx)
        v = torch.zeros(s.size(0), 3, self.hidden_dim, device=s.device)
        
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch_idx)
        i, j = edge_index
        edge_vec = pos[j] - pos[i]
        edge_dist = torch.norm(edge_vec, dim=1)
        edge_vec = edge_vec / (edge_dist.unsqueeze(-1) + 1e-8)
        
        edge_rbf = self.rbf(edge_dist)
        edge_cutoff = self.cutoff_fn(edge_dist)
        
        for layer in self.layers:
            s, v = layer(s, v, edge_index, edge_rbf, edge_cutoff, edge_vec)
        
        s_mol = scatter(s, batch_idx, dim=0, reduce='add')
        fused = self.readout(s_mol, mol_fp)
        
        energy = self.energy_head(fused)
        gap = self.gap_head(fused)
        dipole = self.dipole_head(v, batch_idx)
        npa = self.npa_head(s)
        
        if self.atom_ref is not None:
            ref_energy = self.atom_ref.get_reference_energy_batch(z, batch_idx)
            energy = energy + ref_energy
        
        return {'energy': energy, 'gap': gap, 'dipole': dipole, 'npa': npa}


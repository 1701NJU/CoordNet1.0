import os
import sys
import argparse
import yaml
from datetime import datetime
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coordnet import CoordNet, TmQMDataset, AtomRefCalculator


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_loss(pred, batch, loss_weights, criterion):
    losses = {}
    device = pred['energy'].device
    batch_size = batch.y_energy.size(0)
    num_atoms = torch.bincount(batch.batch, minlength=batch_size).float()
    
    pred_e_per_atom = pred['energy'] / num_atoms
    true_e_per_atom = batch.y_energy.squeeze(-1) / num_atoms
    losses['energy'] = criterion(pred_e_per_atom, true_e_per_atom)
    losses['gap'] = criterion(pred['gap'], batch.y_gap.squeeze(-1))
    losses['dipole'] = criterion(pred['dipole'], batch.y_dipole.squeeze(-1))
    
    if batch.y_npa is not None and pred['npa'] is not None:
        valid_mask = ~torch.isnan(batch.y_npa)
        if valid_mask.any():
            losses['npa'] = criterion(pred['npa'][valid_mask], batch.y_npa[valid_mask])
        else:
            losses['npa'] = torch.tensor(0.0, device=device)
    else:
        losses['npa'] = torch.tensor(0.0, device=device)
    
    total_loss = sum(loss_weights.get(k, 0.0) * v for k, v in losses.items())
    return total_loss, losses


def train_epoch(model, loader, optimizer, criterion, loss_weights, device, max_grad_norm):
    model.train()
    total_loss, n_samples = 0, 0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss, _ = compute_loss(pred, batch, loss_weights, criterion)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n_samples += batch.num_graphs
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / n_samples


@torch.no_grad()
def evaluate(model, loader, criterion, loss_weights, device):
    model.eval()
    total_loss, n_samples = 0, 0
    metrics = {'energy': {'pred': [], 'true': [], 'n_atoms': []}, 'gap': {'pred': [], 'true': []}, 'dipole': {'pred': [], 'true': []}}
    
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        loss, _ = compute_loss(pred, batch, loss_weights, criterion)
        total_loss += loss.item() * batch.num_graphs
        n_samples += batch.num_graphs
        
        batch_size = batch.y_energy.size(0)
        num_atoms = torch.bincount(batch.batch, minlength=batch_size)
        
        metrics['energy']['pred'].extend(pred['energy'].cpu().numpy())
        metrics['energy']['true'].extend(batch.y_energy.squeeze(-1).cpu().numpy())
        metrics['energy']['n_atoms'].extend(num_atoms.cpu().numpy())
        metrics['gap']['pred'].extend(pred['gap'].cpu().numpy())
        metrics['gap']['true'].extend(batch.y_gap.squeeze(-1).cpu().numpy())
        metrics['dipole']['pred'].extend(pred['dipole'].cpu().numpy())
        metrics['dipole']['true'].extend(batch.y_dipole.squeeze(-1).cpu().numpy())
    
    mae = {}
    pred_e = np.array(metrics['energy']['pred'])
    true_e = np.array(metrics['energy']['true'])
    n_atoms = np.array(metrics['energy']['n_atoms'])
    mae['energy'] = np.mean(np.abs(pred_e - true_e))
    mae['energy_per_atom'] = np.mean(np.abs(pred_e - true_e) / n_atoms) * 1000
    
    for task in ['gap', 'dipole']:
        mae[task] = np.mean(np.abs(np.array(metrics[task]['pred']) - np.array(metrics[task]['true'])))
    
    return total_loss / n_samples, mae


def main(args):
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config['logging']['log_dir'], f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    writer = SummaryWriter(run_dir)
    
    print("Loading dataset...")
    dataset = TmQMDataset(root=config['data']['root'], fp_radius=config['data']['fp_radius'], fp_bits=config['data']['fp_bits'])
    print(f"Dataset size: {len(dataset)}")
    
    train_idx, val_idx, test_idx = dataset.get_split_indices(
        split_type=config['data']['split_type'],
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        seed=config['data']['seed']
    )
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    train_dataset = dataset[torch.tensor(train_idx)]
    val_dataset = dataset[torch.tensor(val_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]
    
    print("Fitting AtomRef...")
    atom_ref = AtomRefCalculator()
    train_z_list = [train_dataset[i].z.numpy() for i in range(len(train_dataset))]
    train_charges = np.array([train_dataset[i].charge.item() for i in range(len(train_dataset))])
    train_energies = np.array([train_dataset[i].y_energy.item() for i in range(len(train_dataset))])
    atom_ref.fit(train_z_list, train_charges, train_energies)
    atom_ref.save(os.path.join(run_dir, 'atom_ref.npz'))
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4)
    
    model = CoordNet(
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_rbf=config['model']['num_rbf'],
        cutoff=config['model']['cutoff'],
        num_elements=config['model']['num_elements'],
        num_charges=config['model']['num_charges'],
        fp_dim=config['data']['fp_bits'],
        atom_ref=atom_ref
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['training']['scheduler']['factor'], patience=config['training']['scheduler']['patience'], min_lr=config['training']['scheduler']['min_lr'])
    criterion = nn.L1Loss()
    loss_weights = config['loss_weights']
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, config['training']['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['training']['epochs']}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, loss_weights, device, config['training']['max_grad_norm'])
        val_loss, val_mae = evaluate(model, val_loader, criterion, loss_weights, device)
        scheduler.step(val_loss)
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        for task, mae_val in val_mae.items():
            writer.add_scalar(f'MAE/{task}', mae_val, epoch)
        
        print(f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, Energy MAE: {val_mae['energy']:.2f} eV, Gap MAE: {val_mae['gap']*27.21:.3f} eV")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_loss': val_loss, 'val_mae': val_mae}, os.path.join(run_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= config['training']['patience']:
                print(f"Early stopping at epoch {epoch}")
                break
    
    checkpoint = torch.load(os.path.join(run_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_mae = evaluate(model, test_loader, criterion, loss_weights, device)
    
    print("\n" + "=" * 50)
    print("Test Results")
    print(f"Energy MAE: {test_mae['energy']:.2f} eV ({test_mae['energy_per_atom']:.1f} meV/atom)")
    print(f"Gap MAE: {test_mae['gap']*27.21:.3f} eV")
    print(f"Dipole MAE: {test_mae['dipole']:.3f} D")
    
    with open(os.path.join(run_dir, 'results.yaml'), 'w') as f:
        yaml.dump({'test_loss': test_loss, 'test_mae': {k: float(v) for k, v in test_mae.items()}}, f)
    
    writer.close()
    print(f"\nResults saved to {run_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    main(parser.parse_args())


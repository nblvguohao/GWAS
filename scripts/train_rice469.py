#!/usr/bin/env python
"""
Train PlantHGNN on rice469 dataset
Complete training pipeline with real data
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import json
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.models.plant_hgnn import PlantHGNN
from src.training.losses import RegressionLoss
from src.training.metrics import compute_metrics


class Rice469DataLoader:
    """DataLoader for rice469 dataset"""
    
    def __init__(self, data_dir, split_indices, batch_size=32, shuffle=True):
        self.data_dir = Path(data_dir)
        self.split_indices = split_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Load data
        self.snp_data = torch.from_numpy(np.load(self.data_dir / "genotype_onehot.npy")).float()
        self.phenotype = torch.from_numpy(np.load(self.data_dir / "phenotype_scaled.npy")).float()
        
        # Load graph data
        network_dir = self.data_dir / "networks"
        ppi_data = torch.load(network_dir / "ppi_network.pt", weights_only=False)
        go_data = torch.load(network_dir / "go_network.pt", weights_only=False)
        kegg_data = torch.load(network_dir / "kegg_network.pt", weights_only=False)
        
        self.graph_data = {
            'node_features': ppi_data.x,
            'edge_index_list': [ppi_data.edge_index, go_data.edge_index, kegg_data.edge_index],
            'edge_weight_list': [ppi_data.edge_attr, go_data.edge_attr, kegg_data.edge_attr],
            'random_walk_features': torch.load(network_dir / "random_walk_features.pt", weights_only=False),
            'pagerank_scores': torch.load(network_dir / "pagerank_scores.pt", weights_only=False),
            'gene_set_matrix': torch.load(network_dir / "gene_set_matrix.pt", weights_only=False)
        }
    
    def __iter__(self):
        indices = self.split_indices.copy()
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            
            snp_batch = self.snp_data[batch_indices]
            pheno_batch = self.phenotype[batch_indices]
            
            yield snp_batch, self.graph_data, pheno_batch
    
    def __len__(self):
        return (len(self.split_indices) + self.batch_size - 1) // self.batch_size


def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    n_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for snp_batch, graph_data, pheno_batch in pbar:
        snp_batch = snp_batch.to(device)
        pheno_batch = pheno_batch.to(device)
        
        # Move graph data to device
        graph_data_device = {
            'node_features': graph_data['node_features'].to(device),
            'edge_index_list': [ei.to(device) for ei in graph_data['edge_index_list']],
            'edge_weight_list': [ew.to(device) for ew in graph_data['edge_weight_list']],
            'random_walk_features': graph_data['random_walk_features'].to(device),
            'pagerank_scores': graph_data['pagerank_scores'].to(device),
            'gene_set_matrix': graph_data['gene_set_matrix'].to(device)
        }
        
        optimizer.zero_grad()
        
        predictions = model(snp_batch, graph_data_device)
        loss = loss_fn(predictions, pheno_batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / n_batches


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for snp_batch, graph_data, pheno_batch in dataloader:
            snp_batch = snp_batch.to(device)
            pheno_batch = pheno_batch.to(device)
            
            # Move graph data to device
            graph_data_device = {
                'node_features': graph_data['node_features'].to(device),
                'edge_index_list': [ei.to(device) for ei in graph_data['edge_index_list']],
                'edge_weight_list': [ew.to(device) for ew in graph_data['edge_weight_list']],
                'random_walk_features': graph_data['random_walk_features'].to(device),
                'pagerank_scores': graph_data['pagerank_scores'].to(device),
                'gene_set_matrix': graph_data['gene_set_matrix'].to(device)
            }
            
            predictions = model(snp_batch, graph_data_device)
            
            all_preds.append(predictions.cpu())
            all_targets.append(pheno_batch.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    metrics = compute_metrics(
        all_targets.numpy(),
        all_preds.numpy(),
        metrics=['pearson', 'spearman', 'mse', 'mae']
    )
    
    return metrics, all_preds, all_targets


def main():
    print("="*80)
    print("PlantHGNN Training on Rice469 Dataset")
    print("="*80)
    
    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load metadata
    data_dir = Path("data/processed/rice469")
    
    with open(data_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    with open(data_dir / "split.json", 'r') as f:
        splits = json.load(f)
    
    with open(data_dir / "networks/network_metadata.json", 'r') as f:
        network_metadata = json.load(f)
    
    print("\n" + "="*80)
    print("Dataset Information")
    print("="*80)
    print(f"Samples: {metadata['n_samples']}")
    print(f"SNPs: {metadata['n_snps']}")
    print(f"Traits: {metadata['n_traits']}")
    print(f"Trait names: {', '.join(metadata['trait_names'])}")
    print(f"Genes: {network_metadata['n_genes']}")
    print(f"Gene sets: {network_metadata['n_gene_sets']}")
    print(f"\nSplit:")
    print(f"  Train: {len(splits['train'])} samples")
    print(f"  Val: {len(splits['val'])} samples")
    print(f"  Test: {len(splits['test'])} samples")
    
    # Create dataloaders
    print("\n" + "="*80)
    print("Creating DataLoaders")
    print("="*80)
    
    train_loader = Rice469DataLoader(data_dir, splits['train'], batch_size=32, shuffle=True)
    val_loader = Rice469DataLoader(data_dir, splits['val'], batch_size=32, shuffle=False)
    test_loader = Rice469DataLoader(data_dir, splits['test'], batch_size=32, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print("\n" + "="*80)
    print("Creating PlantHGNN Model")
    print("="*80)
    
    model = PlantHGNN(
        n_snps=metadata['n_snps'],
        n_genes=network_metadata['n_genes'],
        n_gene_sets=network_metadata['n_gene_sets'],
        n_traits=metadata['n_traits'],
        d_model=128,
        n_transformer_layers=8,
        n_attnres_blocks=8,
        n_gcn_layers=2,
        n_views=3,
        dropout=0.2,
        use_heterogeneous=False,
        use_attnres=True,
        use_functional_embed=True,
        use_structural_encode=True
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
    print(f"Model size: {n_params * 4 / 1e6:.1f} MB")
    
    # Setup training
    print("\n" + "="*80)
    print("Training Configuration")
    print("="*80)
    
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = CosineAnnealingLR(optimizer, T_max=20)
    loss_fn = RegressionLoss(loss_type='mse')
    
    print("Optimizer: AdamW (lr=0.001, weight_decay=0.0001)")
    print("Scheduler: CosineAnnealingLR (T_max=20)")
    print("Loss: MSE")
    print("Epochs: 20")
    print("Early stopping patience: 10")
    
    # Training loop
    print("\n" + "="*80)
    print("Training")
    print("="*80)
    
    best_val_pcc = -1
    patience_counter = 0
    patience = 10
    
    history = {
        'train_loss': [],
        'val_metrics': [],
        'test_metrics': None
    }
    
    for epoch in range(1, 21):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/20")
        print(f"{'='*80}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        
        # Validate
        val_metrics, _, _ = evaluate(model, val_loader, device)
        
        # Update scheduler
        scheduler.step()
        
        # Calculate average PCC
        val_pcc_list = [float(x) for x in val_metrics['pearson']]
        val_pcc = np.mean(val_pcc_list)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_metrics'].append(val_metrics)
        
        # Print metrics
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val PCC (avg): {val_pcc:.4f}")
        print(f"Val PCC (per trait): {val_pcc_list}")
        print(f"Val MSE: {val_metrics['mse']:.4f}")
        print(f"Val MAE: {val_metrics['mae']:.4f}")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Early stopping
        if val_pcc > best_val_pcc:
            best_val_pcc = val_pcc
            patience_counter = 0
            torch.save(model.state_dict(), data_dir / "best_model.pt")
            print(f"✓ New best model saved (PCC={best_val_pcc:.4f})")
        else:
            patience_counter += 1
            print(f"⚠ No improvement ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\n⚠ Early stopping triggered at epoch {epoch}")
                break
    
    # Test evaluation
    print("\n" + "="*80)
    print("Final Test Evaluation")
    print("="*80)
    
    # Load best model
    model.load_state_dict(torch.load(data_dir / "best_model.pt"))
    
    test_metrics, test_preds, test_targets = evaluate(model, test_loader, device)
    
    test_pcc_list = [float(x) for x in test_metrics['pearson']]
    test_pcc = np.mean(test_pcc_list)
    
    history['test_metrics'] = test_metrics
    
    print(f"\nTest Results:")
    print(f"  PCC (avg): {test_pcc:.4f}")
    print(f"  PCC (per trait): {test_pcc_list}")
    print(f"  Spearman: {[float(x) for x in test_metrics['spearman']]}")
    print(f"  MSE: {test_metrics['mse']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    
    # Save results
    print("\n" + "="*80)
    print("Saving Results")
    print("="*80)
    
    results = {
        'metadata': metadata,
        'network_metadata': network_metadata,
        'model_config': {
            'n_snps': metadata['n_snps'],
            'n_genes': network_metadata['n_genes'],
            'n_gene_sets': network_metadata['n_gene_sets'],
            'n_traits': metadata['n_traits'],
            'd_model': 128,
            'n_transformer_layers': 8,
            'n_attnres_blocks': 8,
            'n_params': n_params
        },
        'training_config': {
            'optimizer': 'AdamW',
            'lr': 0.001,
            'weight_decay': 0.0001,
            'scheduler': 'CosineAnnealingLR',
            'batch_size': 32,
            'epochs_trained': epoch,
            'early_stopping_patience': patience
        },
        'best_val_pcc': float(best_val_pcc),
        'test_pcc': float(test_pcc),
        'test_metrics': {
            'pearson': test_pcc_list,
            'spearman': [float(x) for x in test_metrics['spearman']],
            'mse': float(test_metrics['mse']),
            'mae': float(test_metrics['mae'])
        },
        'train_loss_history': history['train_loss']
    }
    
    with open(data_dir / "training_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {data_dir / 'training_results.json'}")
    print(f"✓ Best model saved to: {data_dir / 'best_model.pt'}")
    
    # Summary
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Best validation PCC: {best_val_pcc:.4f}")
    print(f"Test PCC: {test_pcc:.4f}")
    print(f"Epochs trained: {epoch}")
    print(f"Model parameters: {n_params:,}")
    
    return 0


if __name__ == '__main__':
    exit(main())

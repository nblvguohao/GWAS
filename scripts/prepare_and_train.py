#!/usr/bin/env python
"""
End-to-end script for data preparation and training
Creates synthetic data for testing the complete pipeline
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

from src.models.plant_hgnn import PlantHGNN
from src.training.losses import RegressionLoss
from src.training.metrics import compute_metrics


def create_synthetic_data(n_samples=100, n_snps=500, n_genes=500, n_traits=3, output_dir='data/synthetic'):
    """Create synthetic data for testing"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating synthetic data...")
    print(f"  Samples: {n_samples}")
    print(f"  SNPs: {n_snps}")
    print(f"  Genes: {n_genes}")
    print(f"  Traits: {n_traits}")
    
    # SNP data (one-hot encoded)
    snp_data = torch.zeros(n_samples, n_snps, 3)
    genotypes = torch.randint(0, 3, (n_samples, n_snps))
    for i in range(n_samples):
        for j in range(n_snps):
            snp_data[i, j, genotypes[i, j]] = 1.0
    
    # Phenotype data (with some correlation to genotypes)
    # Create genetic effects
    genetic_effects = torch.randn(n_snps, n_traits) * 0.1
    genetic_values = torch.zeros(n_samples, n_traits)
    
    for i in range(n_samples):
        for t in range(n_traits):
            genetic_values[i, t] = (genotypes[i].float() * genetic_effects[:, t]).sum()
    
    # Add environmental noise
    phenotype = genetic_values + torch.randn(n_samples, n_traits) * 0.5
    
    # Standardize phenotypes
    phenotype = (phenotype - phenotype.mean(0)) / phenotype.std(0)
    
    # Graph data (PPI, GO, KEGG networks)
    node_features = torch.randn(n_genes, 64)
    
    # Create 3 random networks
    edge_index_list = []
    for _ in range(3):
        n_edges = n_genes * 5  # Average degree of 10
        edge_index = torch.randint(0, n_genes, (2, n_edges))
        edge_index_list.append(edge_index)
    
    # Random walk features
    random_walk_features = torch.randn(n_genes, 10)
    pagerank_scores = torch.rand(n_genes, 1)
    pagerank_scores = pagerank_scores / pagerank_scores.sum()
    
    # Gene set matrix (100 gene sets)
    n_gene_sets = 100
    gene_set_matrix = (torch.rand(n_genes, n_gene_sets) > 0.9).float()
    
    # Save data
    torch.save(snp_data, output_dir / 'snp_data.pt')
    torch.save(phenotype, output_dir / 'phenotype.pt')
    
    graph_data = {
        'node_features': node_features,
        'edge_index_list': edge_index_list,
        'random_walk_features': random_walk_features,
        'pagerank_scores': pagerank_scores,
        'gene_set_matrix': gene_set_matrix
    }
    
    torch.save(graph_data, output_dir / 'graph_data.pt')
    
    # Create train/val/test splits
    indices = torch.randperm(n_samples)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    splits = {
        'train': indices[:train_size].tolist(),
        'val': indices[train_size:train_size+val_size].tolist(),
        'test': indices[train_size+val_size:].tolist()
    }
    
    with open(output_dir / 'splits.json', 'w') as f:
        json.dump(splits, f)
    
    metadata = {
        'n_samples': n_samples,
        'n_snps': n_snps,
        'n_genes': n_genes,
        'n_traits': n_traits,
        'n_gene_sets': n_gene_sets
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Synthetic data saved to {output_dir}")
    print(f"  Train: {len(splits['train'])} samples")
    print(f"  Val: {len(splits['val'])} samples")
    print(f"  Test: {len(splits['test'])} samples")
    
    return metadata


class SimpleDataLoader:
    """Simple DataLoader for synthetic data"""
    
    def __init__(self, snp_data, phenotype, graph_data, indices, batch_size=32, shuffle=True):
        self.snp_data = snp_data
        self.phenotype = phenotype
        self.graph_data = graph_data
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self):
        indices = self.indices.copy()
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            
            snp_batch = self.snp_data[batch_indices]
            pheno_batch = self.phenotype[batch_indices]
            
            yield snp_batch, self.graph_data, pheno_batch
    
    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for snp_batch, graph_data, pheno_batch in dataloader:
        snp_batch = snp_batch.to(device)
        pheno_batch = pheno_batch.to(device)
        
        # Move graph data to device
        graph_data_device = {
            'node_features': graph_data['node_features'].to(device),
            'edge_index_list': [ei.to(device) for ei in graph_data['edge_index_list']],
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
    
    return metrics


def main():
    print("="*60)
    print("PlantHGNN End-to-End Training Pipeline")
    print("="*60)
    
    # Configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Step 1: Create synthetic data
    print("\n" + "="*60)
    print("Step 1: Creating Synthetic Data")
    print("="*60)
    
    metadata = create_synthetic_data(
        n_samples=200,
        n_snps=1000,
        n_genes=1000,
        n_traits=3
    )
    
    # Step 2: Load data
    print("\n" + "="*60)
    print("Step 2: Loading Data")
    print("="*60)
    
    data_dir = Path('data/synthetic')
    
    snp_data = torch.load(data_dir / 'snp_data.pt')
    phenotype = torch.load(data_dir / 'phenotype.pt')
    graph_data = torch.load(data_dir / 'graph_data.pt')
    
    with open(data_dir / 'splits.json', 'r') as f:
        splits = json.load(f)
    
    print(f"Loaded data:")
    print(f"  SNP shape: {snp_data.shape}")
    print(f"  Phenotype shape: {phenotype.shape}")
    print(f"  Graph nodes: {graph_data['node_features'].shape[0]}")
    print(f"  Networks: {len(graph_data['edge_index_list'])}")
    
    # Step 3: Create dataloaders
    print("\n" + "="*60)
    print("Step 3: Creating DataLoaders")
    print("="*60)
    
    train_loader = SimpleDataLoader(
        snp_data, phenotype, graph_data,
        splits['train'], batch_size=32, shuffle=True
    )
    
    val_loader = SimpleDataLoader(
        snp_data, phenotype, graph_data,
        splits['val'], batch_size=32, shuffle=False
    )
    
    test_loader = SimpleDataLoader(
        snp_data, phenotype, graph_data,
        splits['test'], batch_size=32, shuffle=False
    )
    
    print(f"DataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Step 4: Create model
    print("\n" + "="*60)
    print("Step 4: Creating PlantHGNN Model")
    print("="*60)
    
    model = PlantHGNN(
        n_snps=metadata['n_snps'],
        n_genes=metadata['n_genes'],
        n_gene_sets=metadata['n_gene_sets'],
        n_traits=metadata['n_traits'],
        d_model=128,
        n_transformer_layers=8,
        n_attnres_blocks=8,
        n_gcn_layers=2,
        n_views=3,
        dropout=0.2,
        use_heterogeneous=False,  # Simplified for testing
        use_attnres=True,
        use_functional_embed=True,
        use_structural_encode=True
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model created:")
    print(f"  Total parameters: {n_params:,}")
    print(f"  Model size: {n_params * 4 / 1e6:.1f} MB")
    
    # Step 5: Setup training
    print("\n" + "="*60)
    print("Step 5: Setup Training")
    print("="*60)
    
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    loss_fn = RegressionLoss(loss_type='mse')
    
    print("Training configuration:")
    print(f"  Optimizer: AdamW (lr=0.001)")
    print(f"  Scheduler: CosineAnnealingLR")
    print(f"  Loss: MSE")
    print(f"  Epochs: 10")
    
    # Step 6: Training loop
    print("\n" + "="*60)
    print("Step 6: Training for 10 Epochs")
    print("="*60)
    
    best_val_pcc = -1
    history = {
        'train_loss': [],
        'val_metrics': []
    }
    
    for epoch in range(10):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        
        # Update scheduler
        scheduler.step()
        
        # Track best model
        val_pcc = np.mean([float(x) for x in val_metrics['pearson']])
        if val_pcc > best_val_pcc:
            best_val_pcc = val_pcc
            torch.save(model.state_dict(), 'data/synthetic/best_model.pt')
        
        history['train_loss'].append(train_loss)
        history['val_metrics'].append(val_metrics)
        
        # Print progress
        print(f"\nEpoch {epoch+1}/10:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val PCC: {val_pcc:.4f}")
        print(f"  Val MSE: {val_metrics['mse']:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Step 7: Test evaluation
    print("\n" + "="*60)
    print("Step 7: Final Test Evaluation")
    print("="*60)
    
    # Load best model
    model.load_state_dict(torch.load('data/synthetic/best_model.pt'))
    
    test_metrics = evaluate(model, test_loader, device)
    
    print("\nTest Results:")
    print(f"  Pearson: {test_metrics['pearson']}")
    print(f"  Spearman: {test_metrics['spearman']}")
    print(f"  MSE: {test_metrics['mse']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    
    # Step 8: Analyze results
    print("\n" + "="*60)
    print("Step 8: Training Analysis")
    print("="*60)
    
    print("\nTraining Loss Progression:")
    for i, loss in enumerate(history['train_loss']):
        print(f"  Epoch {i+1}: {loss:.4f}")
    
    loss_decrease = history['train_loss'][0] - history['train_loss'][-1]
    loss_decrease_pct = (loss_decrease / history['train_loss'][0]) * 100
    
    print(f"\nLoss Decrease: {loss_decrease:.4f} ({loss_decrease_pct:.1f}%)")
    
    if loss_decrease > 0:
        print("✅ Training loss decreased - model is learning!")
    else:
        print("⚠️ Training loss did not decrease - check configuration")
    
    # Save training history
    with open('data/synthetic/training_history.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_json = {
            'train_loss': history['train_loss'],
            'val_metrics': [
                {k: v.tolist() if isinstance(v, np.ndarray) else v 
                 for k, v in m.items()}
                for m in history['val_metrics']
            ]
        }
        json.dump(history_json, f, indent=2)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nResults saved to: data/synthetic/")
    print(f"  - best_model.pt")
    print(f"  - training_history.json")
    
    return test_metrics


if __name__ == '__main__':
    main()

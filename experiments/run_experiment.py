#!/usr/bin/env python
"""
Main experiment script for PlantHGNN
Runs training and evaluation on specified datasets
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import logging
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.plant_hgnn import PlantHGNN
from src.training.trainer import Trainer
from src.training.losses import RegressionLoss
from src.training.metrics import compute_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(config, n_snps, n_genes, n_gene_sets, n_traits):
    """Create PlantHGNN model from config"""
    model_config = config['model']
    
    model = PlantHGNN(
        n_snps=n_snps,
        n_genes=n_genes,
        n_gene_sets=n_gene_sets,
        n_traits=n_traits,
        d_model=model_config['d_model'],
        n_transformer_layers=model_config['n_transformer_layers'],
        n_attnres_blocks=model_config['n_attnres_blocks'],
        n_gcn_layers=model_config['n_gcn_layers'],
        n_views=model_config['n_views'],
        dropout=model_config['dropout'],
        use_heterogeneous=model_config['use_heterogeneous'],
        use_attnres=model_config['use_attnres'],
        use_functional_embed=model_config['use_functional_embed'],
        use_structural_encode=model_config['use_structural_encode']
    )
    
    return model


def create_optimizer(model, config):
    """Create optimizer from config"""
    train_config = config['training']
    
    if train_config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_config['lr'],
            weight_decay=train_config['weight_decay']
        )
    elif train_config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config['lr'],
            weight_decay=train_config['weight_decay']
        )
    elif train_config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=train_config['lr'],
            weight_decay=train_config['weight_decay'],
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {train_config['optimizer']}")
    
    return optimizer


def create_scheduler(optimizer, config):
    """Create learning rate scheduler from config"""
    train_config = config['training']
    
    if train_config.get('scheduler') == 'cosine_annealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_config['max_epochs']
        )
    elif train_config.get('scheduler') == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
    else:
        scheduler = None
    
    return scheduler


def run_experiment(config, dataset_name, output_dir, seed=42):
    """
    Run a single experiment
    
    Args:
        config: Configuration dict
        dataset_name: Name of dataset (e.g., 'rice469')
        output_dir: Output directory
        seed: Random seed
    
    Returns:
        Test metrics
    """
    set_seed(seed)
    
    logger.info(f"Running experiment: {dataset_name}, seed={seed}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # TODO: Load actual data
    # For now, create dummy data for testing
    logger.info("Loading data...")
    
    # Dummy dimensions (replace with actual data loading)
    n_snps = 1000
    n_genes = 500
    n_gene_sets = 100
    n_traits = 3
    batch_size = config['training']['batch_size']
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config, n_snps, n_genes, n_gene_sets, n_traits)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Create loss function
    loss_fn = RegressionLoss(loss_type=config['training']['loss_type'])
    
    # Create trainer
    device = config['hardware']['device'] if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        metrics=config['evaluation']['metrics'],
        early_stopping_patience=config['training']['early_stopping_patience'],
        scheduler=scheduler
    )
    
    # TODO: Create data loaders from actual data
    # For now, skip training and just save config
    
    logger.info("Experiment setup complete!")
    logger.info(f"Results will be saved to: {output_dir}")
    
    # Save experiment info
    experiment_info = {
        'dataset': dataset_name,
        'seed': seed,
        'model_params': sum(p.numel() for p in model.parameters()),
        'config': config
    }
    
    with open(output_dir / 'experiment_info.json', 'w') as f:
        json.dump(experiment_info, f, indent=2)
    
    return {'status': 'configured', 'output_dir': str(output_dir)}


def main():
    parser = argparse.ArgumentParser(description='Run PlantHGNN experiment')
    parser.add_argument('--config', required=True, help='Path to config YAML file')
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g., rice469)')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    
    args = parser.parse_args()
    
    # Set GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        logger.info(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        logger.info("CUDA not available, using CPU")
    
    # Load config
    config = load_config(args.config)
    
    # Run experiment
    results = run_experiment(
        config=config,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    logger.info("Experiment complete!")
    logger.info(f"Results: {results}")


if __name__ == '__main__':
    main()

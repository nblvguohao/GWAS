"""
Training module for PlantHGNN
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import json

from .metrics import compute_metrics, MetricsTracker
from .losses import RegressionLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for PlantHGNN
    
    Args:
        model: Model to train
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to train on
        metrics: List of metrics to compute
        early_stopping_patience: Patience for early stopping
    """
    
    def __init__(
        self,
        model,
        optimizer,
        loss_fn=None,
        device='cuda',
        metrics=['pearson', 'mse', 'mae', 'ndcg'],
        early_stopping_patience=20,
        scheduler=None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn or RegressionLoss()
        self.device = device
        self.metrics = metrics
        self.early_stopping_patience = early_stopping_patience
        self.scheduler = scheduler
        
        self.metrics_tracker = MetricsTracker()
        self.best_val_metric = -np.inf
        self.patience_counter = 0
        self.best_model_state = None
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        for batch in tqdm(train_loader, desc="Training", leave=False):
            snp_data = batch['snp_data'].to(self.device)
            targets = batch['phenotype'].to(self.device)
            graph_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch['graph_data'].items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(snp_data, graph_data)
            
            # Compute loss
            loss = self.loss_fn(predictions, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            all_predictions.append(predictions.detach().cpu())
            all_targets.append(targets.detach().cpu())
        
        # Aggregate predictions
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics
        metrics_dict = compute_metrics(all_targets, all_predictions, self.metrics)
        metrics_dict['loss'] = total_loss / len(train_loader)
        
        return metrics_dict
    
    def evaluate(self, data_loader, split='val'):
        """Evaluate on validation or test set"""
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {split}", leave=False):
                snp_data = batch['snp_data'].to(self.device)
                targets = batch['phenotype'].to(self.device)
                graph_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in batch['graph_data'].items()}
                
                # Forward pass
                predictions = self.model(snp_data, graph_data)
                
                # Compute loss
                loss = self.loss_fn(predictions, targets)
                
                total_loss += loss.item()
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Aggregate predictions
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics
        metrics_dict = compute_metrics(all_targets, all_predictions, self.metrics)
        metrics_dict['loss'] = total_loss / len(data_loader)
        
        return metrics_dict, all_predictions, all_targets
    
    def train(self, train_loader, val_loader, n_epochs, save_dir=None):
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Number of epochs
            save_dir: Directory to save checkpoints
        
        Returns:
            Best validation metrics
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting training for {n_epochs} epochs...")
        
        for epoch in range(n_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{n_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.metrics_tracker.update('train', train_metrics)
            
            # Validate
            val_metrics, _, _ = self.evaluate(val_loader, split='val')
            self.metrics_tracker.update('val', val_metrics)
            
            # Log metrics
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"PCC: {np.mean(train_metrics['pearson']):.4f}")
            logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                       f"PCC: {np.mean(val_metrics['pearson']):.4f}")
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Early stopping check
            current_val_metric = np.mean(val_metrics['pearson'])
            
            if current_val_metric > self.best_val_metric:
                self.best_val_metric = current_val_metric
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
                
                if save_dir:
                    checkpoint_path = save_dir / 'best_model.pt'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_metrics': val_metrics,
                    }, checkpoint_path)
                    logger.info(f"Saved best model to {checkpoint_path}")
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        # Get best validation metrics
        best_epoch = self.metrics_tracker.get_best_epoch('val', 'pearson', 'max')
        best_val_metrics = self.metrics_tracker.get_metric_at_epoch(best_epoch, 'val')
        
        logger.info(f"\nTraining complete!")
        logger.info(f"Best epoch: {best_epoch + 1}")
        logger.info(f"Best val PCC: {np.mean(best_val_metrics['pearson']):.4f}")
        
        return best_val_metrics
    
    def test(self, test_loader, save_dir=None):
        """
        Test the model
        
        Args:
            test_loader: Test data loader
            save_dir: Directory to save results
        
        Returns:
            Test metrics and predictions
        """
        logger.info("Evaluating on test set...")
        
        test_metrics, predictions, targets = self.evaluate(test_loader, split='test')
        self.metrics_tracker.update('test', test_metrics)
        
        logger.info(f"Test - Loss: {test_metrics['loss']:.4f}, "
                   f"PCC: {np.mean(test_metrics['pearson']):.4f}")
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save predictions
            results = {
                'predictions': predictions.numpy().tolist(),
                'targets': targets.numpy().tolist(),
                'metrics': {k: v if not isinstance(v, list) else v 
                           for k, v in test_metrics.items()}
            }
            
            with open(save_dir / 'test_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Saved test results to {save_dir}")
        
        return test_metrics, predictions, targets
    
    def save_checkpoint(self, path, epoch, additional_info=None):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
            'patience_counter': self.patience_counter,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_metric = checkpoint.get('best_val_metric', -np.inf)
        self.patience_counter = checkpoint.get('patience_counter', 0)
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded checkpoint from {path}")
        
        return checkpoint.get('epoch', 0)

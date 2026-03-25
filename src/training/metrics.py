"""
Evaluation metrics for PlantHGNN
Implements PCC, NDCG, MSE, MAE, and statistical tests
"""

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from scipy.stats import wilcoxon
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score


def compute_pearson_correlation(y_true, y_pred):
    """
    Compute Pearson correlation coefficient
    
    Args:
        y_true: True values (n_samples,) or (n_samples, n_traits)
        y_pred: Predicted values (n_samples,) or (n_samples, n_traits)
    
    Returns:
        PCC value or list of PCC values for each trait
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    if y_true.ndim == 1:
        corr, _ = pearsonr(y_true, y_pred)
        return corr
    else:
        # Multiple traits
        correlations = []
        for i in range(y_true.shape[1]):
            corr, _ = pearsonr(y_true[:, i], y_pred[:, i])
            correlations.append(corr)
        return correlations


def compute_spearman_correlation(y_true, y_pred):
    """Compute Spearman rank correlation coefficient"""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    if y_true.ndim == 1:
        corr, _ = spearmanr(y_true, y_pred)
        return corr
    else:
        correlations = []
        for i in range(y_true.shape[1]):
            corr, _ = spearmanr(y_true[:, i], y_pred[:, i])
            correlations.append(corr)
        return correlations


def compute_ndcg(y_true, y_pred, k=10):
    """
    Compute Normalized Discounted Cumulative Gain at k
    Useful for ranking-based breeding selection
    
    Args:
        y_true: True values
        y_pred: Predicted values
        k: Top-k to consider
    
    Returns:
        NDCG@k score
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
        y_pred = y_pred.reshape(1, -1)
    
    try:
        ndcg = ndcg_score(y_true, y_pred, k=k)
    except:
        # Handle edge cases
        ndcg = 0.0
    
    return ndcg


def compute_mse(y_true, y_pred):
    """Compute Mean Squared Error"""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    return mean_squared_error(y_true, y_pred)


def compute_mae(y_true, y_pred):
    """Compute Mean Absolute Error"""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    return mean_absolute_error(y_true, y_pred)


def compute_metrics(y_true, y_pred, metrics=['pearson', 'spearman', 'mse', 'mae', 'ndcg']):
    """
    Compute all specified metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        metrics: List of metric names
    
    Returns:
        Dictionary of metric values
    """
    results = {}
    
    if 'pearson' in metrics or 'pcc' in metrics:
        results['pearson'] = compute_pearson_correlation(y_true, y_pred)
    
    if 'spearman' in metrics:
        results['spearman'] = compute_spearman_correlation(y_true, y_pred)
    
    if 'mse' in metrics:
        results['mse'] = compute_mse(y_true, y_pred)
    
    if 'mae' in metrics:
        results['mae'] = compute_mae(y_true, y_pred)
    
    if 'ndcg' in metrics:
        results['ndcg@10'] = compute_ndcg(y_true, y_pred, k=10)
    
    return results


def wilcoxon_test(results_a, results_b):
    """
    Perform Wilcoxon signed-rank test
    
    Args:
        results_a: Results from method A (list of scores)
        results_b: Results from method B (list of scores)
    
    Returns:
        p-value
    """
    results_a = np.array(results_a)
    results_b = np.array(results_b)
    
    if len(results_a) != len(results_b):
        raise ValueError("Results must have same length")
    
    if len(results_a) < 6:
        # Not enough samples for Wilcoxon test
        return 1.0
    
    try:
        _, p_value = wilcoxon(results_a, results_b)
    except:
        p_value = 1.0
    
    return p_value


def format_metric_with_significance(mean, std, p_value):
    """
    Format metric with significance stars
    
    Args:
        mean: Mean value
        std: Standard deviation
        p_value: P-value from statistical test
    
    Returns:
        Formatted string
    """
    if p_value < 0.001:
        stars = '***'
    elif p_value < 0.01:
        stars = '**'
    elif p_value < 0.05:
        stars = '*'
    else:
        stars = ''
    
    return f"{mean:.4f} ± {std:.4f}{stars}"


def aggregate_cv_results(fold_results):
    """
    Aggregate results from cross-validation folds
    
    Args:
        fold_results: List of result dicts from each fold
    
    Returns:
        Dictionary with mean and std for each metric
    """
    aggregated = {}
    
    # Get all metric names
    metric_names = fold_results[0].keys()
    
    for metric in metric_names:
        values = [fold[metric] for fold in fold_results]
        
        # Handle multi-trait case
        if isinstance(values[0], list):
            # Average across folds for each trait
            n_traits = len(values[0])
            trait_means = []
            trait_stds = []
            
            for trait_idx in range(n_traits):
                trait_values = [v[trait_idx] for v in values]
                trait_means.append(np.mean(trait_values))
                trait_stds.append(np.std(trait_values))
            
            aggregated[f'{metric}_mean'] = trait_means
            aggregated[f'{metric}_std'] = trait_stds
        else:
            # Single value
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
    
    return aggregated


class MetricsTracker:
    """Track metrics during training"""
    
    def __init__(self):
        self.history = {
            'train': [],
            'val': [],
            'test': []
        }
    
    def update(self, split, metrics_dict):
        """Update metrics for a split"""
        self.history[split].append(metrics_dict)
    
    def get_best_epoch(self, split='val', metric='pearson', mode='max'):
        """Get epoch with best metric value"""
        if len(self.history[split]) == 0:
            return 0
        
        values = [m[metric] for m in self.history[split]]
        
        # Handle multi-trait case (take mean)
        values = [np.mean(v) if isinstance(v, list) else v for v in values]
        
        if mode == 'max':
            best_epoch = np.argmax(values)
        else:
            best_epoch = np.argmin(values)
        
        return best_epoch
    
    def get_metric_at_epoch(self, epoch, split='val'):
        """Get metrics at specific epoch"""
        if epoch >= len(self.history[split]):
            return None
        return self.history[split][epoch]
    
    def plot_history(self, metric='pearson', save_path=None):
        """Plot training history"""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for split in ['train', 'val', 'test']:
            if len(self.history[split]) > 0:
                values = [m.get(metric, 0) for m in self.history[split]]
                # Handle multi-trait
                values = [np.mean(v) if isinstance(v, list) else v for v in values]
                ax.plot(values, label=split)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} over Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def test_metrics():
    """Test metrics implementation"""
    print("Testing metrics...")
    
    # Generate dummy data
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.5
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)
    
    print("Single trait metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Multi-trait
    y_true_multi = np.random.randn(100, 3)
    y_pred_multi = y_true_multi + np.random.randn(100, 3) * 0.5
    
    metrics_multi = compute_metrics(y_true_multi, y_pred_multi)
    
    print("\nMulti-trait metrics:")
    for name, value in metrics_multi.items():
        if isinstance(value, list):
            print(f"  {name}: {[f'{v:.4f}' for v in value]}")
        else:
            print(f"  {name}: {value:.4f}")
    
    # Test Wilcoxon
    results_a = [0.5, 0.6, 0.55, 0.58, 0.62, 0.59]
    results_b = [0.45, 0.52, 0.48, 0.51, 0.54, 0.50]
    p_value = wilcoxon_test(results_a, results_b)
    print(f"\nWilcoxon test p-value: {p_value:.4f}")
    
    print("\nMetrics test passed!")


if __name__ == '__main__':
    test_metrics()

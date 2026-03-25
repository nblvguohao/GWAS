"""
Loss functions for PlantHGNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionLoss(nn.Module):
    """
    Regression loss for genomic prediction
    Supports MSE, MAE, and Huber loss
    """
    
    def __init__(self, loss_type='mse', huber_delta=1.0):
        super().__init__()
        self.loss_type = loss_type
        self.huber_delta = huber_delta
    
    def forward(self, predictions, targets):
        """
        Compute loss
        
        Args:
            predictions: Predicted values (batch_size, n_traits)
            targets: True values (batch_size, n_traits)
        
        Returns:
            Loss value
        """
        if self.loss_type == 'mse':
            return F.mse_loss(predictions, targets)
        elif self.loss_type == 'mae':
            return F.l1_loss(predictions, targets)
        elif self.loss_type == 'huber':
            return F.huber_loss(predictions, targets, delta=self.huber_delta)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with learnable task weights
    Useful for multi-trait prediction
    """
    
    def __init__(self, n_tasks, loss_type='mse'):
        super().__init__()
        self.n_tasks = n_tasks
        self.loss_type = loss_type
        
        # Learnable task weights (log variance)
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))
    
    def forward(self, predictions, targets):
        """
        Compute weighted multi-task loss
        
        Loss = Σ_i (1 / (2 * σ_i^2)) * L_i + log(σ_i)
        where σ_i^2 = exp(log_var_i)
        """
        losses = []
        
        for i in range(self.n_tasks):
            pred_i = predictions[:, i]
            target_i = targets[:, i]
            
            if self.loss_type == 'mse':
                loss_i = F.mse_loss(pred_i, target_i, reduction='mean')
            elif self.loss_type == 'mae':
                loss_i = F.l1_loss(pred_i, target_i, reduction='mean')
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
            
            # Weighted by uncertainty
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss_i + self.log_vars[i]
            
            losses.append(weighted_loss)
        
        return sum(losses)
    
    def get_task_weights(self):
        """Get task weights (inverse of variance)"""
        return torch.exp(-self.log_vars).detach()


class RankingLoss(nn.Module):
    """
    Ranking loss for breeding selection
    Encourages correct ranking of individuals
    """
    
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, predictions, targets):
        """
        Pairwise ranking loss
        
        For pairs (i, j) where target_i > target_j,
        penalize if pred_i <= pred_j + margin
        """
        batch_size = predictions.shape[0]
        
        # Expand to pairwise comparisons
        pred_i = predictions.unsqueeze(1)  # (batch, 1, n_traits)
        pred_j = predictions.unsqueeze(0)  # (1, batch, n_traits)
        
        target_i = targets.unsqueeze(1)
        target_j = targets.unsqueeze(0)
        
        # Compute ranking violations
        # If target_i > target_j, we want pred_i > pred_j + margin
        should_rank_higher = (target_i > target_j).float()
        ranking_violation = F.relu(self.margin - (pred_i - pred_j))
        
        loss = (should_rank_higher * ranking_violation).mean()
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss: regression + ranking
    """
    
    def __init__(self, regression_weight=1.0, ranking_weight=0.1, loss_type='mse'):
        super().__init__()
        self.regression_loss = RegressionLoss(loss_type)
        self.ranking_loss = RankingLoss()
        self.regression_weight = regression_weight
        self.ranking_weight = ranking_weight
    
    def forward(self, predictions, targets):
        """Compute combined loss"""
        reg_loss = self.regression_loss(predictions, targets)
        rank_loss = self.ranking_loss(predictions, targets)
        
        total_loss = (self.regression_weight * reg_loss + 
                     self.ranking_weight * rank_loss)
        
        return total_loss, {'regression': reg_loss.item(), 'ranking': rank_loss.item()}


def test_losses():
    """Test loss functions"""
    print("Testing loss functions...")
    
    batch_size = 16
    n_traits = 3
    
    predictions = torch.randn(batch_size, n_traits)
    targets = torch.randn(batch_size, n_traits)
    
    # Test regression loss
    reg_loss = RegressionLoss(loss_type='mse')
    loss_val = reg_loss(predictions, targets)
    print(f"MSE loss: {loss_val.item():.4f}")
    
    # Test multi-task loss
    mt_loss = MultiTaskLoss(n_traits)
    loss_val = mt_loss(predictions, targets)
    print(f"Multi-task loss: {loss_val.item():.4f}")
    print(f"Task weights: {mt_loss.get_task_weights()}")
    
    # Test ranking loss
    rank_loss = RankingLoss()
    loss_val = rank_loss(predictions, targets)
    print(f"Ranking loss: {loss_val.item():.4f}")
    
    # Test combined loss
    combined = CombinedLoss()
    total_loss, loss_dict = combined(predictions, targets)
    print(f"Combined loss: {total_loss.item():.4f}")
    print(f"  Regression: {loss_dict['regression']:.4f}")
    print(f"  Ranking: {loss_dict['ranking']:.4f}")
    
    print("\nLoss functions test passed!")


if __name__ == '__main__':
    test_losses()

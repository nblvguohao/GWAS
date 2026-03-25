"""
DNNGP: Deep Neural Network for Genomic Prediction
Simplified re-implementation of the DNNGP architecture.

Reference: Liu et al., DNNGP: A deep neural network-based method for
           genomic prediction using multi-omics data (2023)

Core architecture: FC → BN → ReLU (stacked layers) with Dropout
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr

from .base import BaselineModel


class _DNNGPNet(nn.Module):
    def __init__(self, n_snps: int, hidden_dims=(512, 256, 128, 64),
                 dropout: float = 0.3):
        super().__init__()
        layers = []
        in_dim = n_snps
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DNNGP(BaselineModel):
    """
    DNNGP: Deep Neural Network for Genomic Prediction

    Args:
        hidden_dims: Tuple of hidden layer dimensions
        dropout: Dropout rate
        lr: Learning rate
        weight_decay: L2 regularization
        batch_size: Training batch size
        max_epochs: Maximum training epochs
        patience: Early stopping patience
        device: 'cuda' or 'cpu'
    """

    def __init__(
        self,
        hidden_dims=(512, 256, 128, 64),
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 200,
        patience: int = 20,
        device: str = 'auto',
    ):
        super().__init__(name="DNNGP")
        self.hidden_dims  = hidden_dims
        self.dropout      = dropout
        self.lr           = lr
        self.weight_decay = weight_decay
        self.batch_size   = batch_size
        self.max_epochs   = max_epochs
        self.patience     = patience
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        n_snps = X_train.shape[1]
        self.model = _DNNGPNet(n_snps, self.hidden_dims, self.dropout).to(self.device)

        opt = optim.AdamW(self.model.parameters(),
                          lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.max_epochs, eta_min=1e-5
        )

        X_tr = torch.tensor(X_train, dtype=torch.float32)
        y_tr = torch.tensor(y_train, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_tr, y_tr),
                            batch_size=self.batch_size, shuffle=True,
                            drop_last=False)

        best_val_pcc = -1.0
        best_state   = None
        no_improve   = 0

        for epoch in range(self.max_epochs):
            self.model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                loss = nn.MSELoss()(self.model(xb), yb)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
            scheduler.step()

            if X_val is not None:
                val_pred = self.predict(X_val)
                val_pcc, _ = pearsonr(y_val, val_pred)
                if val_pcc > best_val_pcc:
                    best_val_pcc = val_pcc
                    best_state   = {k: v.cpu().clone()
                                    for k, v in self.model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.is_fitted = True

    def predict(self, X_test, **kwargs):
        if self.model is None:
            raise RuntimeError("Call fit() first")
        self.model.eval()
        Xt = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(Xt).cpu().numpy()
        return preds

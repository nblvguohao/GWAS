"""
NetGP: Network-based Genomic Prediction baseline.

Simplified re-implementation following the NetGP paper (PBJ 2025):
  - 2-layer GCN on a gene-level network
  - Gene features = mean additive coding of mapped SNPs
  - Global pooling → FC head for trait prediction

Key difference from PlantHGNN:
  - Single homogeneous network (no multi-view, no attention fusion)
  - No Transformer layers, no AttnRes
  - Simpler architecture = direct comparison point

Reference: Zhang et al., NetGP: Network-guided Genomic Prediction (PBJ 2025)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr

from .base import BaselineModel


class _GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x, adj):
        # x: (batch, n_genes, in_dim), adj: (n_genes, n_genes)
        x = self.linear(x)
        x = torch.matmul(adj.unsqueeze(0), x)
        return x


class _NetGPNet(nn.Module):
    def __init__(self, n_snps: int, n_genes: int,
                 d_hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        # SNP pathway (for samples with no gene mapping)
        self.snp_fc = nn.Sequential(
            nn.Linear(n_snps, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # GCN pathway
        self.gcn1 = _GCNLayer(1, d_hidden)
        self.bn1  = nn.BatchNorm1d(d_hidden)
        self.gcn2 = _GCNLayer(d_hidden, d_hidden)
        self.drop = nn.Dropout(dropout)

        # Fusion + head
        self.fusion = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(d_hidden, 1)
        self.n_genes = n_genes

    def forward(self, snp, gene_feat, adj):
        # snp: (B, n_snps)
        # gene_feat: (B, n_genes) → (B, n_genes, 1)
        # adj: (n_genes, n_genes)
        h_snp = self.snp_fc(snp)  # (B, d)

        x_g = gene_feat.unsqueeze(-1)  # (B, n_genes, 1)
        B, G, _ = x_g.shape

        h = self.gcn1(x_g, adj)                        # (B, G, d)
        h = h.reshape(-1, h.size(-1))
        h = self.bn1(h)
        h = h.reshape(B, G, -1)
        h = F.relu(h)
        h = self.drop(h)
        h = self.gcn2(h, adj)                          # (B, G, d)
        h_gcn = h.mean(dim=1)                          # (B, d)

        h = self.fusion(torch.cat([h_snp, h_gcn], dim=-1))
        return self.head(h).squeeze(-1)


class NetGP(BaselineModel):
    """
    NetGP: Network-guided Genomic Prediction

    Args:
        d_hidden: Hidden dimension for GCN
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
        d_hidden: int = 128,
        dropout: float = 0.2,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 200,
        patience: int = 20,
        device: str = 'auto',
    ):
        super().__init__(name="NetGP")
        self.d_hidden     = d_hidden
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
        self._adj = None

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            gene_train=None, gene_val=None, adj=None, **kwargs):
        """
        Args:
            X_train:    (n_train, n_snps)
            y_train:    (n_train,)
            gene_train: (n_train, n_genes) gene-level features (optional)
            adj:        (n_genes, n_genes) normalized adjacency (optional)
        """
        n_snps = X_train.shape[1]

        if gene_train is not None and adj is not None:
            n_genes = gene_train.shape[1]
            self._adj = torch.tensor(adj, dtype=torch.float32).to(self.device)
            use_gcn = True
        else:
            # Fallback: create dummy gene data (pure SNP model)
            use_gcn = False
            n_genes = 1

        self.use_gcn = use_gcn
        self.model = _NetGPNet(n_snps, n_genes, self.d_hidden,
                               self.dropout).to(self.device)

        opt = optim.AdamW(self.model.parameters(),
                          lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.max_epochs, eta_min=1e-5
        )

        X_tr = torch.tensor(X_train, dtype=torch.float32)
        y_tr = torch.tensor(y_train, dtype=torch.float32)
        if use_gcn:
            g_tr = torch.tensor(gene_train, dtype=torch.float32)
            ds   = TensorDataset(X_tr, y_tr, g_tr)
        else:
            dummy_g = torch.zeros(len(y_tr), n_genes)
            ds = TensorDataset(X_tr, y_tr, dummy_g)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        best_pcc   = -1.0
        best_state = None
        no_improve = 0

        adj_dev = self._adj if use_gcn else torch.zeros(1, 1).to(self.device)

        for epoch in range(self.max_epochs):
            self.model.train()
            for xb, yb, gb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                gb = gb.to(self.device)
                pred = self.model(xb, gb, adj_dev)
                loss = nn.MSELoss()(pred, yb)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
            scheduler.step()

            if X_val is not None:
                preds = self.predict(X_val, gene_val)
                pcc, _ = pearsonr(y_val, preds)
                if pcc > best_pcc:
                    best_pcc   = pcc
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.is_fitted = True

    def predict(self, X_test, gene_test=None, **kwargs):
        if self.model is None:
            raise RuntimeError("Call fit() first")
        self.model.eval()
        Xt = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        adj_dev = self._adj if (self.use_gcn and self._adj is not None) \
                  else torch.zeros(1, 1).to(self.device)
        if gene_test is not None and self.use_gcn:
            Gt = torch.tensor(gene_test, dtype=torch.float32).to(self.device)
        else:
            Gt = torch.zeros(len(Xt), 1).to(self.device)
        with torch.no_grad():
            preds = self.model(Xt, Gt, adj_dev).cpu().numpy()
        return preds

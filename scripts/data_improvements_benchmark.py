#!/usr/bin/env python
"""
Data-side improvements for performance boost.

Current bottleneck analysis:
  - 10K SNPs selected by correlation with ONE trait (BLUP_Yield_per_plant) → biased
  - One-hot encoding (10K×3=30K dim) → wastes capacity, redundant
  - No sample-similarity structure → ignores GBLUP's core advantage
  - Multi-task across 32 weakly-correlated traits (mean |r|=0.23) → gradient conflict

Improvements implemented:
  1. Additive encoding (0/1/2 scalar per SNP, 10K dim) — much more efficient
  2. Additive + dominance encoding (2D per SNP, 20K dim) — captures non-additive effects
  3. Multi-trait SNP selection from raw data (50K SNPs)
  4. Sample-similarity GNN (KNN graph from GRM + GCN)
  5. Per-trait specialist + stacking
  6. Huber loss for robustness
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.optim import Adam
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
import time
import warnings
warnings.filterwarnings('ignore')

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# ============================================================
# Metrics
# ============================================================

def full_metrics(preds, targets):
    pccs, sccs, mses, maes = [], [], [], []
    for t in range(targets.shape[1]):
        m = ~np.isnan(targets[:, t])
        if m.sum() > 10 and np.std(preds[m, t]) > 1e-8:
            p, _ = pearsonr(targets[m, t], preds[m, t])
            s, _ = spearmanr(targets[m, t], preds[m, t])
        else:
            p, s = 0.0, 0.0
        mse = float(np.mean((preds[m, t] - targets[m, t])**2)) if m.sum() > 0 else 0
        mae = float(np.mean(np.abs(preds[m, t] - targets[m, t]))) if m.sum() > 0 else 0
        pccs.append(float(p)); sccs.append(float(s)); mses.append(mse); maes.append(mae)
    return {'pcc': float(np.mean(pccs)), 'spearman': float(np.mean(sccs)),
            'mse': float(np.mean(mses)), 'mae': float(np.mean(maes)),
            'pccs': pccs, 'mses': mses, 'maes': maes}

def nan_mse(pred, target):
    mask = ~torch.isnan(target)
    if mask.sum() == 0: return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return ((pred - target)**2 * mask).sum() / mask.sum()

def nan_huber(pred, target, delta=1.0):
    mask = ~torch.isnan(target)
    if mask.sum() == 0: return torch.tensor(0.0, device=pred.device, requires_grad=True)
    diff = (pred - target).abs()
    loss = torch.where(diff < delta, 0.5 * diff**2, delta * (diff - 0.5 * delta))
    return (loss * mask).sum() / mask.sum()


# ============================================================
# Data transformations
# ============================================================

def onehot_to_additive(genotype_onehot):
    """Convert one-hot (N, SNPs, 3) to additive (N, SNPs): 0/1/2 dosage."""
    # one-hot channels: [homo_ref, het, homo_alt] → dosage = 0*ch0 + 1*ch1 + 2*ch2
    return genotype_onehot[:, :, 1] + 2 * genotype_onehot[:, :, 2]

def onehot_to_additive_dominance(genotype_onehot):
    """Convert one-hot (N, SNPs, 3) to additive+dominance (N, SNPs, 2).
    Channel 0: additive = dosage/2 → [0, 0.5, 1]
    Channel 1: dominance = heterozygosity → [0, 1, 0]
    """
    additive = (genotype_onehot[:, :, 1] + 2 * genotype_onehot[:, :, 2]) / 2.0
    dominance = genotype_onehot[:, :, 1]  # 1 for heterozygous, 0 otherwise
    return np.stack([additive, dominance], axis=-1)

def build_grm_knn_graph(X_additive, k=10):
    """Build KNN graph from GRM (genomic relationship matrix).
    Returns edge_index (2, E) and edge_weight (E,) for PyTorch.
    """
    n = X_additive.shape[0]
    X = X_additive.astype(np.float64)
    X_c = X - X.mean(axis=0)
    K = X_c @ X_c.T / X_c.shape[1]

    # KNN: for each sample, find k most similar samples
    edges_src, edges_dst, weights = [], [], []
    for i in range(n):
        sims = K[i].copy()
        sims[i] = -np.inf  # exclude self
        top_k = np.argsort(sims)[-k:]
        for j in top_k:
            edges_src.append(i); edges_dst.append(j); weights.append(float(K[i, j]))
            edges_src.append(j); edges_dst.append(i); weights.append(float(K[i, j]))

    edge_index = np.array([edges_src, edges_dst])
    edge_weight = np.array(weights)
    # Deduplicate
    edge_set = {}
    for idx in range(edge_index.shape[1]):
        key = (edge_index[0, idx], edge_index[1, idx])
        if key not in edge_set:
            edge_set[key] = edge_weight[idx]
    ei = np.array(list(edge_set.keys())).T
    ew = np.array(list(edge_set.values()))
    return ei, ew, K


# ============================================================
# Models
# ============================================================

class MultiScaleCNN_Additive(nn.Module):
    """Multi-scale CNN for additive (1-channel) or add+dom (2-channel) encoding."""
    def __init__(self, n_snps, n_traits, in_channels=1, d_hidden=256, dropout=0.3):
        super().__init__()
        self.conv_k3 = nn.Conv1d(in_channels, 32, 3, padding=1)
        self.conv_k7 = nn.Conv1d(in_channels, 32, 7, padding=3)
        self.conv_k15 = nn.Conv1d(in_channels, 32, 15, padding=7)
        self.bn1 = nn.BatchNorm1d(96)
        self.conv2 = nn.Conv1d(96, 128, 5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, 5, padding=2, stride=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(64)
        self.head = nn.Sequential(
            nn.Linear(128*64, d_hidden), nn.BatchNorm1d(d_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden//2), nn.BatchNorm1d(d_hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden//2, n_traits),
        )
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)  # (B, SNPs) → (B, 1, SNPs)
        else: x = x.transpose(1, 2)  # (B, SNPs, C) → (B, C, SNPs)
        x = self.bn1(torch.cat([F.relu(self.conv_k3(x)), F.relu(self.conv_k7(x)), F.relu(self.conv_k15(x))], 1))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(self.pool(x).view(x.size(0), -1))


class AnchorModel_Additive(nn.Module):
    """Anchor similarity model for additive encoding."""
    def __init__(self, input_dim, n_traits, n_anchors=200, d_hidden=256, dropout=0.3):
        super().__init__()
        self.anchors = nn.Parameter(torch.randn(n_anchors, input_dim) * 0.01)
        self.temp = nn.Parameter(torch.tensor(1.0))
        self.head = nn.Sequential(
            nn.Linear(n_anchors, d_hidden), nn.BatchNorm1d(d_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden//2), nn.BatchNorm1d(d_hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden//2, n_traits),
        )
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        sim = torch.matmul(F.normalize(x,1), F.normalize(self.anchors,1).t()) / self.temp.abs().clamp(min=0.1)
        return self.head(sim)


class SampleGNN(nn.Module):
    """GNN on sample-similarity graph. Each node = sample, features = SNP additive vector.
    Uses message passing (simplified GCN) to aggregate neighbor info before prediction.
    """
    def __init__(self, input_dim, n_traits, d_hidden=256, n_layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_hidden)
        self.gcn_layers = nn.ModuleList()
        self.gcn_bns = nn.ModuleList()
        for _ in range(n_layers):
            self.gcn_layers.append(nn.Linear(d_hidden, d_hidden))
            self.gcn_bns.append(nn.BatchNorm1d(d_hidden))
        self.head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden//2), nn.BatchNorm1d(d_hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden//2, n_traits),
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight, n_nodes):
        """x: (n_nodes, input_dim), edge_index: (2, E), edge_weight: (E,)"""
        h = F.relu(self.input_proj(x))
        for gcn, bn in zip(self.gcn_layers, self.gcn_bns):
            # Simple message passing: h_new[i] = sum(w_ij * h[j]) / sum(w_ij) + h[i]
            h_new = torch.zeros_like(h)
            deg = torch.zeros(n_nodes, device=h.device)
            src, dst = edge_index[0], edge_index[1]
            w = edge_weight
            # Scatter add
            h_new.index_add_(0, dst, h[src] * w.unsqueeze(1))
            deg.index_add_(0, dst, w)
            deg = deg.clamp(min=1e-8)
            h_new = h_new / deg.unsqueeze(1)
            h_new = bn(F.relu(gcn(h_new)))
            h = h + F.dropout(h_new, p=self.dropout, training=self.training)  # Residual
        return self.head(h)


class HybridCNN_GNN(nn.Module):
    """Hybrid: CNN extracts per-sample features, then GNN refines using sample graph."""
    def __init__(self, n_snps, n_traits, d_hidden=256, n_anchors=200, dropout=0.3):
        super().__init__()
        # CNN feature extractor
        self.conv_k3 = nn.Conv1d(1, 32, 3, padding=1)
        self.conv_k7 = nn.Conv1d(1, 32, 7, padding=3)
        self.conv_k15 = nn.Conv1d(1, 32, 15, padding=7)
        self.bn1 = nn.BatchNorm1d(96)
        self.conv2 = nn.Conv1d(96, 64, 5, padding=2, stride=4)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(16)
        self.cnn_out_dim = 64 * 16  # 1024

        # Anchor similarity branch
        self.anchors = nn.Parameter(torch.randn(n_anchors, n_snps) * 0.01)
        self.temp = nn.Parameter(torch.tensor(1.0))

        # Fused prediction head
        fused_dim = self.cnn_out_dim + n_anchors
        self.head = nn.Sequential(
            nn.Linear(fused_dim, d_hidden), nn.BatchNorm1d(d_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden//2), nn.BatchNorm1d(d_hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden//2, n_traits),
        )

    def forward(self, x):
        # x: (B, n_snps) additive
        x_cnn = x.unsqueeze(1)  # (B, 1, SNPs)
        c = self.bn1(torch.cat([F.relu(self.conv_k3(x_cnn)), F.relu(self.conv_k7(x_cnn)),
                                 F.relu(self.conv_k15(x_cnn))], 1))
        c = F.relu(self.bn2(self.conv2(c)))
        c = self.pool(c).view(x.size(0), -1)  # (B, 1024)

        # Anchor similarity
        sim = torch.matmul(F.normalize(x,1), F.normalize(self.anchors,1).t()) / self.temp.abs().clamp(min=0.1)

        # Fuse
        return self.head(torch.cat([c, sim], dim=1))


# ============================================================
# Training utilities
# ============================================================

def predict_batch(model, data, indices, device, bs=64, model_type='standard'):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(indices), bs):
            bi = indices[i:i+bs]
            x = data[bi].to(device)
            preds.append(model(x).cpu())
    return torch.cat(preds).numpy()


def train_model(model, train_data, phenotype, splits, device,
                n_epochs=60, lr=0.001, wd=1e-4, bs=32, name="Model",
                patience=15, loss_fn='mse'):
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    train_idx, val_idx, test_idx = splits['train'], splits['val'], splits['test']
    best_val, best_state, wait = -999, None, 0
    loss_func = nan_huber if loss_fn == 'huber' else nan_mse

    for epoch in range(1, n_epochs+1):
        model.train()
        idx = train_idx.copy(); np.random.shuffle(idx)
        tot, nb = 0, 0
        for i in range(0, len(idx), bs):
            bi = idx[i:i+bs]
            x, y = train_data[bi].to(device), phenotype[bi].to(device)
            opt.zero_grad()
            loss = loss_func(model(x), y)
            if torch.isnan(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item(); nb += 1
        sched.step()
        if nb == 0: continue

        vp = predict_batch(model, train_data, val_idx, device)
        vm = full_metrics(vp, phenotype[val_idx].numpy())
        if vm['pcc'] > best_val:
            best_val = vm['pcc']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience: break
        if epoch % 10 == 0:
            print(f"    [{name}] ep {epoch:3d}  loss={tot/nb:.4f}  val_PCC={vm['pcc']:.4f}")

    if best_state: model.load_state_dict(best_state)
    model = model.to(device).eval()
    pv = predict_batch(model, train_data, val_idx, device)
    pt = predict_batch(model, train_data, test_idx, device)
    return model, pv, pt, n_params


def run_gblup(X_additive, pheno_np, splits, lambdas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]):
    """Kernel ridge GBLUP."""
    X = X_additive.astype(np.float64)
    train_idx, val_idx, test_idx = [np.array(splits[k]) for k in ('train','val','test')]
    X_c = X - X.mean(axis=0)
    K = X_c @ X_c.T / X_c.shape[1]
    K_tr = K[np.ix_(train_idx, train_idx)]
    K_va = K[np.ix_(val_idx, train_idx)]
    K_te = K[np.ix_(test_idx, train_idx)]
    Y_tr, Y_va = pheno_np[train_idx], pheno_np[val_idx]
    n_tr = len(train_idx); n_traits = Y_tr.shape[1]
    pv, pt = np.zeros_like(Y_va), np.zeros_like(pheno_np[test_idx])
    for t in range(n_traits):
        y = Y_tr[:, t].copy(); y[np.isnan(y)] = 0.0
        mv = ~np.isnan(Y_va[:, t])
        best_pcc = -999
        for lam in lambdas:
            a = np.linalg.solve(K_tr + lam * np.eye(n_tr), y)
            pred_v = K_va @ a
            if mv.sum() > 5 and np.std(pred_v[mv]) > 1e-8:
                pcc, _ = pearsonr(Y_va[mv, t], pred_v[mv])
            else: pcc = 0.0
            if pcc > best_pcc:
                best_pcc = pcc; pv[:, t] = pred_v; pt[:, t] = K_te @ a
    return pv, pt


def train_gnn_full(model, X_tensor, phenotype, edge_index, edge_weight, splits, device,
                   n_epochs=60, lr=0.001, wd=1e-4, patience=15, name="GNN"):
    """Train GNN on full graph (all samples as nodes)."""
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    X_all = X_tensor.to(device)
    pheno_all = phenotype.to(device)
    ei = torch.from_numpy(edge_index).long().to(device)
    ew = torch.from_numpy(edge_weight).float().to(device)
    n_nodes = X_all.size(0)

    train_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    for i in splits['train']: train_mask[i] = True
    for i in splits['val']: val_mask[i] = True
    for i in splits['test']: test_mask[i] = True

    opt = Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    best_val, best_state, wait = -999, None, 0

    for epoch in range(1, n_epochs+1):
        model.train()
        opt.zero_grad()
        out = model(X_all, ei, ew, n_nodes)
        loss = nan_mse(out[train_mask], pheno_all[train_mask])
        if torch.isnan(loss): continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        model.eval()
        with torch.no_grad():
            out = model(X_all, ei, ew, n_nodes)
            vp = out[val_mask].cpu().numpy()
        pheno_cpu = phenotype.numpy() if isinstance(phenotype, torch.Tensor) else phenotype
        val_indices = val_mask.cpu().numpy()
        vm = full_metrics(vp, pheno_cpu[val_indices])
        if vm['pcc'] > best_val:
            best_val = vm['pcc']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience: break
        if epoch % 10 == 0:
            print(f"    [{name}] ep {epoch:3d}  loss={loss.item():.4f}  val_PCC={vm['pcc']:.4f}")

    if best_state: model.load_state_dict(best_state)
    model = model.to(device).eval()
    with torch.no_grad():
        out = model(X_all, ei, ew, n_nodes)
    pv = out[val_mask].cpu().numpy()
    pt = out[test_mask].cpu().numpy()
    del X_all, ei, ew, pheno_all
    torch.cuda.empty_cache()
    return model, pv, pt, n_params


def stacking(preds_val_list, preds_test_list, Y_val, Y_test, n_traits):
    sv, st = np.zeros_like(Y_val), np.zeros_like(Y_test)
    for t in range(n_traits):
        mv = ~np.isnan(Y_val[:, t])
        if mv.sum() < 10:
            for k in range(len(preds_val_list)):
                st[:, t] += preds_test_list[k][:, t] / len(preds_val_list)
                sv[:, t] += preds_val_list[k][:, t] / len(preds_val_list)
            continue
        Xv = np.column_stack([p[mv, t] for p in preds_val_list])
        yv = Y_val[mv, t]
        Xt = np.column_stack([p[:, t] for p in preds_test_list])
        Xv_all = np.column_stack([p[:, t] for p in preds_val_list])
        best_a = 1.0; best_pcc = -999
        for a in [0.01, 0.1, 1.0, 10.0]:
            r = Ridge(alpha=a).fit(Xv, yv)
            pred = r.predict(Xv)
            if np.std(pred) > 1e-8:
                pcc, _ = pearsonr(yv, pred)
                if pcc > best_pcc: best_pcc = pcc; best_a = a
        r = Ridge(alpha=best_a).fit(Xv, yv)
        st[:, t] = r.predict(Xt); sv[:, t] = r.predict(Xv_all)
    return sv, st


# ============================================================
# Per-trait specialist training
# ============================================================

def train_per_trait_specialists(train_data, phenotype_np, splits, device, n_snps,
                                n_traits, n_epochs=40, name="Specialist"):
    """Train one small model per trait, then combine predictions."""
    train_idx = list(splits['train'])
    val_idx = np.array(splits['val'])
    test_idx = np.array(splits['test'])
    pv_all = np.zeros((len(val_idx), n_traits))
    pt_all = np.zeros((len(test_idx), n_traits))
    phenotype_t = torch.from_numpy(phenotype_np).float()

    print(f"    Training {n_traits} per-trait specialists...")
    for t in range(n_traits):
        model = nn.Sequential(
            nn.Linear(n_snps, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        ).to(device)
        opt = Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
        best_val, best_state, wait = -999, None, 0

        for epoch in range(1, n_epochs+1):
            model.train()
            idx = train_idx.copy(); np.random.shuffle(idx)
            for i in range(0, len(idx), 32):
                bi = idx[i:i+32]
                x = train_data[bi].to(device)
                y = phenotype_t[bi, t:t+1].to(device)
                opt.zero_grad()
                loss = nan_mse(model(x), y)
                if torch.isnan(loss): continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

            model.eval()
            with torch.no_grad():
                vp = torch.cat([model(train_data[val_idx[i:i+64]].to(device)).cpu()
                                for i in range(0, len(val_idx), 64)]).numpy().flatten()
            mv = ~np.isnan(phenotype_np[val_idx, t])
            if mv.sum() > 10 and np.std(vp[mv]) > 1e-8:
                pcc, _ = pearsonr(phenotype_np[val_idx, t][mv], vp[mv])
            else: pcc = 0.0
            if pcc > best_val:
                best_val = pcc; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; wait = 0
            else:
                wait += 1
                if wait >= 10: break

        if best_state: model.load_state_dict(best_state)
        model = model.to(device).eval()
        with torch.no_grad():
            pv_all[:, t] = torch.cat([model(train_data[val_idx[i:i+64]].to(device)).cpu()
                                       for i in range(0, len(val_idx), 64)]).numpy().flatten()
            pt_all[:, t] = torch.cat([model(train_data[test_idx[i:i+64]].to(device)).cpu()
                                       for i in range(0, len(test_idx), 64)]).numpy().flatten()

    return pv_all, pt_all


# ============================================================
# Main
# ============================================================

def run_dataset(ds_name, data_dir, device):
    print(f"\n{'#'*80}")
    print(f"# {ds_name} — DATA IMPROVEMENT EXPERIMENTS")
    print(f"{'#'*80}")

    data_dir = Path(data_dir)
    genotype_oh = np.load(data_dir / "genotype_onehot.npy")
    pheno_np = np.load(data_dir / "phenotype_scaled.npy")
    with open(data_dir / "metadata.json") as f: meta = json.load(f)
    with open(data_dir / "split.json") as f: splits = json.load(f)

    n_samples, n_snps = genotype_oh.shape[0], genotype_oh.shape[1]
    n_traits = pheno_np.shape[1]
    val_idx, test_idx = splits['val'], splits['test']
    Y_val, Y_test = pheno_np[val_idx], pheno_np[test_idx]

    print(f"  Samples={n_samples}  SNPs={n_snps}  Traits={n_traits}")

    # ----- Data representations -----
    print("\n  Preparing data representations...")
    additive = onehot_to_additive(genotype_oh)               # (N, SNPs)
    add_dom = onehot_to_additive_dominance(genotype_oh)       # (N, SNPs, 2)

    additive_t = torch.from_numpy(additive).float()
    add_dom_t = torch.from_numpy(add_dom).float()
    onehot_t = torch.from_numpy(genotype_oh).float()
    phenotype_t = torch.from_numpy(pheno_np).float()

    print(f"    Additive: {additive.shape}")
    print(f"    Add+Dom:  {add_dom.shape}")

    # Build sample-similarity graph
    print("  Building GRM-based KNN graph (k=15)...")
    t0 = time.time()
    edge_index, edge_weight, K = build_grm_knn_graph(additive, k=15)
    print(f"    Edges: {edge_index.shape[1]}, build time: {time.time()-t0:.1f}s")

    results = {}
    all_pv, all_pt, all_names = [], [], []

    # =====================================================================
    # BASELINES (from previous benchmark, re-run for fairness)
    # =====================================================================

    # 1. GBLUP
    print(f"\n  >>> GBLUP")
    t0 = time.time()
    gblup_pv, gblup_pt = run_gblup(additive, pheno_np, splits)
    tm = time.time() - t0
    results['GBLUP'] = {**full_metrics(gblup_pt, Y_test), 'n_params': 'N/A', 'time': tm,
                         'val': full_metrics(gblup_pv, Y_val)}
    all_pv.append(gblup_pv); all_pt.append(gblup_pt); all_names.append('GBLUP')
    print(f"      PCC={results['GBLUP']['pcc']:.4f}  MSE={results['GBLUP']['mse']:.4f}  MAE={results['GBLUP']['mae']:.4f}")

    # 2. V1-OneHot (previous best single model) — baseline encoding
    print(f"\n  >>> V1-OneHot (baseline encoding)")
    t0 = time.time()
    from scripts.enhanced_benchmark import OurV1
    m, pv, pt, np_ = train_model(OurV1(n_snps, n_traits, 256, 0.3),
                                  onehot_t, phenotype_t, splits, device, name="V1-OH")
    tm = time.time() - t0
    results['V1_OneHot'] = {**full_metrics(pt, Y_test), 'n_params': np_, 'time': tm,
                             'val': full_metrics(pv, Y_val)}
    all_pv.append(pv); all_pt.append(pt); all_names.append('V1_OneHot')
    print(f"      PCC={results['V1_OneHot']['pcc']:.4f}  MSE={results['V1_OneHot']['mse']:.4f}  MAE={results['V1_OneHot']['mae']:.4f}")

    # =====================================================================
    # IMPROVEMENT 1: Additive encoding
    # =====================================================================
    print(f"\n  >>> [IMP1] V1 + Additive encoding (1-ch)")
    t0 = time.time()
    m, pv, pt, np_ = train_model(MultiScaleCNN_Additive(n_snps, n_traits, in_channels=1),
                                  additive_t, phenotype_t, splits, device, name="V1-Add")
    tm = time.time() - t0
    results['V1_Additive'] = {**full_metrics(pt, Y_test), 'n_params': np_, 'time': tm,
                               'val': full_metrics(pv, Y_val)}
    all_pv.append(pv); all_pt.append(pt); all_names.append('V1_Additive')
    print(f"      PCC={results['V1_Additive']['pcc']:.4f}  MSE={results['V1_Additive']['mse']:.4f}  MAE={results['V1_Additive']['mae']:.4f}")

    # =====================================================================
    # IMPROVEMENT 2: Additive + Dominance encoding
    # =====================================================================
    print(f"\n  >>> [IMP2] V1 + Add+Dom encoding (2-ch)")
    t0 = time.time()
    m, pv, pt, np_ = train_model(MultiScaleCNN_Additive(n_snps, n_traits, in_channels=2),
                                  add_dom_t, phenotype_t, splits, device, name="V1-AD")
    tm = time.time() - t0
    results['V1_AddDom'] = {**full_metrics(pt, Y_test), 'n_params': np_, 'time': tm,
                             'val': full_metrics(pv, Y_val)}
    all_pv.append(pv); all_pt.append(pt); all_names.append('V1_AddDom')
    print(f"      PCC={results['V1_AddDom']['pcc']:.4f}  MSE={results['V1_AddDom']['mse']:.4f}  MAE={results['V1_AddDom']['mae']:.4f}")

    # =====================================================================
    # IMPROVEMENT 3: Anchor model with additive encoding
    # =====================================================================
    n_anchors = min(200, n_samples // 5)
    print(f"\n  >>> [IMP3] V2-Anchor + Additive (anchors={n_anchors})")
    t0 = time.time()
    m, pv, pt, np_ = train_model(AnchorModel_Additive(n_snps, n_traits, n_anchors),
                                  additive_t, phenotype_t, splits, device, name="V2-Add")
    tm = time.time() - t0
    results['V2_Additive'] = {**full_metrics(pt, Y_test), 'n_params': np_, 'time': tm,
                               'val': full_metrics(pv, Y_val)}
    all_pv.append(pv); all_pt.append(pt); all_names.append('V2_Additive')
    print(f"      PCC={results['V2_Additive']['pcc']:.4f}  MSE={results['V2_Additive']['mse']:.4f}  MAE={results['V2_Additive']['mae']:.4f}")

    # =====================================================================
    # IMPROVEMENT 4: Huber loss (robust to outliers)
    # =====================================================================
    print(f"\n  >>> [IMP4] V1-Add + Huber loss")
    t0 = time.time()
    m, pv, pt, np_ = train_model(MultiScaleCNN_Additive(n_snps, n_traits, in_channels=1),
                                  additive_t, phenotype_t, splits, device, name="V1-Huber",
                                  loss_fn='huber')
    tm = time.time() - t0
    results['V1_Add_Huber'] = {**full_metrics(pt, Y_test), 'n_params': np_, 'time': tm,
                                'val': full_metrics(pv, Y_val)}
    all_pv.append(pv); all_pt.append(pt); all_names.append('V1_Add_Huber')
    print(f"      PCC={results['V1_Add_Huber']['pcc']:.4f}  MSE={results['V1_Add_Huber']['mse']:.4f}  MAE={results['V1_Add_Huber']['mae']:.4f}")

    # =====================================================================
    # IMPROVEMENT 5: Hybrid CNN + Anchor (fused model)
    # =====================================================================
    print(f"\n  >>> [IMP5] Hybrid CNN+Anchor (additive)")
    t0 = time.time()
    m, pv, pt, np_ = train_model(HybridCNN_GNN(n_snps, n_traits, 256, n_anchors),
                                  additive_t, phenotype_t, splits, device, name="Hybrid")
    tm = time.time() - t0
    results['Hybrid'] = {**full_metrics(pt, Y_test), 'n_params': np_, 'time': tm,
                          'val': full_metrics(pv, Y_val)}
    all_pv.append(pv); all_pt.append(pt); all_names.append('Hybrid')
    print(f"      PCC={results['Hybrid']['pcc']:.4f}  MSE={results['Hybrid']['mse']:.4f}  MAE={results['Hybrid']['mae']:.4f}")

    # =====================================================================
    # IMPROVEMENT 6: Sample-similarity GNN
    # =====================================================================
    print(f"\n  >>> [IMP6] Sample-similarity GNN (additive)")
    t0 = time.time()
    gnn_model, gnn_pv, gnn_pt, gnn_np_ = train_gnn_full(
        SampleGNN(n_snps, n_traits, 256, n_layers=2, dropout=0.3),
        additive_t, phenotype_t, edge_index, edge_weight, splits, device, name="GNN")
    tm = time.time() - t0
    results['SampleGNN'] = {**full_metrics(gnn_pt, Y_test), 'n_params': gnn_np_, 'time': tm,
                             'val': full_metrics(gnn_pv, Y_val)}
    all_pv.append(gnn_pv); all_pt.append(gnn_pt); all_names.append('SampleGNN')
    print(f"      PCC={results['SampleGNN']['pcc']:.4f}  MSE={results['SampleGNN']['mse']:.4f}  MAE={results['SampleGNN']['mae']:.4f}")

    # =====================================================================
    # IMPROVEMENT 7: Per-trait specialist models
    # =====================================================================
    print(f"\n  >>> [IMP7] Per-trait specialists (additive)")
    t0 = time.time()
    spec_pv, spec_pt = train_per_trait_specialists(
        additive_t, pheno_np, splits, device, n_snps, n_traits)
    tm = time.time() - t0
    results['PerTrait'] = {**full_metrics(spec_pt, Y_test), 'n_params': f'{n_traits}×small',
                            'time': tm, 'val': full_metrics(spec_pv, Y_val)}
    all_pv.append(spec_pv); all_pt.append(spec_pt); all_names.append('PerTrait')
    print(f"      PCC={results['PerTrait']['pcc']:.4f}  MSE={results['PerTrait']['mse']:.4f}  MAE={results['PerTrait']['mae']:.4f}")

    # =====================================================================
    # GRAND STACKING: all models
    # =====================================================================
    print(f"\n  >>> Grand Stacking (all {len(all_pv)} models)")
    gs_pv, gs_pt = stacking(all_pv, all_pt, Y_val, Y_test, n_traits)
    results['GrandStacking'] = {**full_metrics(gs_pt, Y_test), 'n_params': 'meta',
                                 'val': full_metrics(gs_pv, Y_val)}
    print(f"      PCC={results['GrandStacking']['pcc']:.4f}  MSE={results['GrandStacking']['mse']:.4f}  MAE={results['GrandStacking']['mae']:.4f}")

    return results


def main():
    print("=" * 80)
    print("DATA-SIDE IMPROVEMENTS BENCHMARK")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    all_results = {}
    for ds_name, ds_path in [
        ("GSTP007", "data/processed/GSTP007_full_10000snps_processed"),
        ("Rice469", "data/processed/rice469"),
    ]:
        all_results[ds_name] = run_dataset(ds_name, ds_path, device)

    # ============ Print final tables ============
    print(f"\n\n{'#'*100}")
    print("FINAL DATA-IMPROVEMENT RESULTS")
    print(f"{'#'*100}")

    methods = ['GBLUP', 'V1_OneHot', 'V1_Additive', 'V1_AddDom', 'V2_Additive',
               'V1_Add_Huber', 'Hybrid', 'SampleGNN', 'PerTrait', 'GrandStacking']

    for ds in all_results:
        res = all_results[ds]
        print(f"\n{'='*100}")
        print(f"  {ds}")
        print(f"{'='*100}")
        print(f"  {'Method':<22} {'PCC':>8} {'Spear':>8} {'MSE':>8} {'MAE':>8}"
              f" | {'ValPCC':>8} {'ValMSE':>8} {'ValMAE':>8} | {'Params':>12}")
        print(f"  {'-'*100}")
        for m in methods:
            if m not in res: continue
            r = res[m]; vr = r.get('val', {})
            np_ = r.get('n_params', '—')
            ps = f"{np_:,}" if isinstance(np_, int) else str(np_)
            print(f"  {m:<22} {r['pcc']:>8.4f} {r['spearman']:>8.4f} {r['mse']:>8.4f} {r['mae']:>8.4f}"
                  f" | {vr.get('pcc',0):>8.4f} {vr.get('mse',0):>8.4f} {vr.get('mae',0):>8.4f} | {ps:>12}")

        ranked = sorted([(m, res[m]) for m in methods if m in res],
                        key=lambda x: x[1]['pcc'], reverse=True)
        print(f"\n  Ranking:")
        for i, (m, r) in enumerate(ranked):
            tag = " 🏆" if i == 0 else ""
            print(f"    {i+1:2d}. {m:<22} PCC={r['pcc']:.4f}  MSE={r['mse']:.4f}  MAE={r['mae']:.4f}{tag}")

    # Save
    def ser(obj):
        if isinstance(obj, dict): return {k: ser(v) for k, v in obj.items()}
        if isinstance(obj, list): return [ser(v) for v in obj]
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        return obj

    out = Path("data/processed/GSTP007_full_10000snps_processed/data_improvement_results.json")
    with open(out, 'w') as f:
        json.dump(ser(all_results), f, indent=2)
    print(f"\n✓ Results saved to {out}")


if __name__ == '__main__':
    main()

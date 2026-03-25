#!/usr/bin/env python
"""
Enhanced benchmark with:
  1. All improvement strategies: V2-400 anchors, V1-deeper, Stacking, 3-way ensemble (V1+V2+GBLUP)
  2. Full metrics: PCC, Spearman, MSE, MAE
  3. Both datasets
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
from sklearn.linear_model import Ridge as SkRidge
import time
import warnings
warnings.filterwarnings('ignore')

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# Metrics (NaN-safe, per-trait then averaged)
# ============================================================

def full_metrics(preds, targets):
    """Return dict with PCC, Spearman, MSE, MAE — all NaN-safe, per-trait averaged."""
    pccs, sccs, mses, maes = [], [], [], []
    per_trait = []
    for t in range(targets.shape[1]):
        m = ~np.isnan(targets[:, t])
        n_valid = m.sum()
        if n_valid > 10 and np.std(preds[m, t]) > 1e-8:
            p, _ = pearsonr(targets[m, t], preds[m, t])
            s, _ = spearmanr(targets[m, t], preds[m, t])
        else:
            p, s = 0.0, 0.0
        if n_valid > 0:
            mse = float(np.mean((preds[m, t] - targets[m, t]) ** 2))
            mae = float(np.mean(np.abs(preds[m, t] - targets[m, t])))
        else:
            mse, mae = 0.0, 0.0
        pccs.append(float(p)); sccs.append(float(s))
        mses.append(mse); maes.append(mae)
        per_trait.append({'pcc': float(p), 'spearman': float(s), 'mse': mse, 'mae': mae})
    return {
        'pcc': float(np.mean(pccs)),
        'spearman': float(np.mean(sccs)),
        'mse': float(np.mean(mses)),
        'mae': float(np.mean(maes)),
        'per_trait': per_trait,
    }


def nan_safe_mse_loss(pred, target):
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return ((pred - target) ** 2 * mask).sum() / mask.sum()


def load_dataset(data_dir):
    data_dir = Path(data_dir)
    g = torch.from_numpy(np.load(data_dir / "genotype_onehot.npy")).float()
    p = torch.from_numpy(np.load(data_dir / "phenotype_scaled.npy")).float()
    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)
    with open(data_dir / "split.json") as f:
        splits = json.load(f)
    return g, p, meta, splits


# ============================================================
# GBLUP (kernel ridge)
# ============================================================

def run_gblup(snp_np, pheno_np, splits, lambdas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]):
    n_samples = snp_np.shape[0]
    X = snp_np.reshape(n_samples, -1).astype(np.float64)
    train_idx, val_idx, test_idx = [np.array(splits[k]) for k in ('train', 'val', 'test')]

    X_c = X - X.mean(axis=0)
    K = X_c @ X_c.T / X_c.shape[1]
    K_tr = K[np.ix_(train_idx, train_idx)]
    K_va = K[np.ix_(val_idx, train_idx)]
    K_te = K[np.ix_(test_idx, train_idx)]
    Y_tr, Y_va, Y_te = pheno_np[train_idx], pheno_np[val_idx], pheno_np[test_idx]
    n_tr = len(train_idx); n_traits = Y_tr.shape[1]

    preds_val = np.zeros_like(Y_va)
    preds_test = np.zeros_like(Y_te)
    for t in range(n_traits):
        y = Y_tr[:, t].copy(); y[np.isnan(y)] = 0.0
        mv = ~np.isnan(Y_va[:, t])
        best_pcc = -999
        for lam in lambdas:
            a = np.linalg.solve(K_tr + lam * np.eye(n_tr), y)
            pv = K_va @ a
            if mv.sum() > 5 and np.std(pv[mv]) > 1e-8:
                pcc, _ = pearsonr(Y_va[mv, t], pv[mv])
            else:
                pcc = 0.0
            if pcc > best_pcc:
                best_pcc = pcc; preds_val[:, t] = pv; preds_test[:, t] = K_te @ a
    return preds_val, preds_test


# ============================================================
# Model definitions
# ============================================================

class MLPBaseline(nn.Module):
    def __init__(self, input_dim, n_traits, hidden=512, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 4), nn.BatchNorm1d(hidden // 4), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 4, n_traits),
        )
    def forward(self, x):
        return self.net(x.reshape(x.size(0), -1))


class DNNGP(nn.Module):
    def __init__(self, n_snps, n_traits, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(3, 16, 5, padding=2), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(32),
        )
        self.head = nn.Sequential(
            nn.Linear(64*32, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_traits),
        )
    def forward(self, x):
        return self.head(self.features(x.transpose(1,2)).view(x.size(0),-1))


class CNNBaseline(nn.Module):
    def __init__(self, n_snps, n_traits, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(3, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(4),
            nn.AdaptiveAvgPool1d(16),
        )
        self.head = nn.Sequential(
            nn.Linear(128*16, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, n_traits),
        )
    def forward(self, x):
        return self.head(self.features(x.transpose(1,2)).view(x.size(0),-1))


class OurV1(nn.Module):
    """Multi-scale CNN."""
    def __init__(self, n_snps, n_traits, d_hidden=256, dropout=0.3):
        super().__init__()
        self.conv_k3 = nn.Conv1d(3, 32, 3, padding=1)
        self.conv_k7 = nn.Conv1d(3, 32, 7, padding=3)
        self.conv_k15 = nn.Conv1d(3, 32, 15, padding=7)
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
        x = x.transpose(1, 2)
        x = self.bn1(torch.cat([F.relu(self.conv_k3(x)), F.relu(self.conv_k7(x)), F.relu(self.conv_k15(x))], 1))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(self.pool(x).view(x.size(0), -1))


class OurV1Deep(nn.Module):
    """Deeper Multi-scale CNN with residual connections."""
    def __init__(self, n_snps, n_traits, d_hidden=256, dropout=0.3):
        super().__init__()
        self.conv_k3 = nn.Conv1d(3, 32, 3, padding=1)
        self.conv_k7 = nn.Conv1d(3, 32, 7, padding=3)
        self.conv_k15 = nn.Conv1d(3, 32, 15, padding=7)
        self.conv_k31 = nn.Conv1d(3, 32, 31, padding=15)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 128, 5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, 5, padding=2, stride=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 128, 3, padding=1, stride=2)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(32)
        self.head = nn.Sequential(
            nn.Linear(128*32, d_hidden), nn.BatchNorm1d(d_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden//2), nn.BatchNorm1d(d_hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden//2, n_traits),
        )
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.bn1(torch.cat([F.relu(self.conv_k3(x)), F.relu(self.conv_k7(x)),
                                 F.relu(self.conv_k15(x)), F.relu(self.conv_k31(x))], 1))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return self.head(self.pool(x).view(x.size(0), -1))


class OurV2(nn.Module):
    """Anchor similarity."""
    def __init__(self, n_snps, n_traits, n_anchors=200, d_hidden=256, dropout=0.3):
        super().__init__()
        self.anchors = nn.Parameter(torch.randn(n_anchors, n_snps*3) * 0.01)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.head = nn.Sequential(
            nn.Linear(n_anchors, d_hidden), nn.BatchNorm1d(d_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden//2), nn.BatchNorm1d(d_hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden//2, n_traits),
        )
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        sim = torch.matmul(F.normalize(x,1), F.normalize(self.anchors,1).t()) / self.temperature.abs().clamp(min=0.1)
        return self.head(sim)


class OurV2Large(nn.Module):
    """Anchor similarity with more anchors + deeper head."""
    def __init__(self, n_snps, n_traits, n_anchors=400, d_hidden=512, dropout=0.3):
        super().__init__()
        self.anchors = nn.Parameter(torch.randn(n_anchors, n_snps*3) * 0.01)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.head = nn.Sequential(
            nn.Linear(n_anchors, d_hidden), nn.BatchNorm1d(d_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden//2), nn.BatchNorm1d(d_hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden//2, d_hidden//4), nn.BatchNorm1d(d_hidden//4), nn.ReLU(), nn.Dropout(dropout*0.5),
            nn.Linear(d_hidden//4, n_traits),
        )
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        sim = torch.matmul(F.normalize(x,1), F.normalize(self.anchors,1).t()) / self.temperature.abs().clamp(min=0.1)
        return self.head(sim)


# ============================================================
# Generic DL trainer — returns model, predictions on val & test
# ============================================================

def predict_batch(model, snp_data, indices, device, bs=64):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(indices), bs):
            bi = indices[i:i+bs]
            preds.append(model(snp_data[bi].to(device)).cpu())
    return torch.cat(preds).numpy()


def train_dl(model, snp_data, phenotype, splits, device,
             n_epochs=60, lr=0.001, wd=1e-4, bs=32, name="Model", patience=15):
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    train_idx, val_idx, test_idx = splits['train'], splits['val'], splits['test']
    best_val, best_state, wait = -999, None, 0

    for epoch in range(1, n_epochs+1):
        model.train()
        idx = train_idx.copy(); np.random.shuffle(idx)
        tot, nb = 0, 0
        for i in range(0, len(idx), bs):
            bi = idx[i:i+bs]
            x, y = snp_data[bi].to(device), phenotype[bi].to(device)
            opt.zero_grad()
            loss = nan_safe_mse_loss(model(x), y)
            if torch.isnan(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if nb == 0: continue

        vp = predict_batch(model, snp_data, val_idx, device)
        vm = full_metrics(vp, phenotype[val_idx].numpy())
        vpcc = vm['pcc']
        if vpcc > best_val:
            best_val = vpcc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience: break
        if epoch % 10 == 0:
            print(f"    [{name}] ep {epoch:3d}  loss={tot/nb:.4f}  val_PCC={vpcc:.4f}")

    if best_state: model.load_state_dict(best_state)
    model = model.to(device).eval()
    preds_val = predict_batch(model, snp_data, val_idx, device)
    preds_test = predict_batch(model, snp_data, test_idx, device)
    return model, preds_val, preds_test, n_params


# ============================================================
# Stacking meta-learner
# ============================================================

def stacking_ensemble(preds_val_list, preds_test_list, Y_val, Y_test, n_traits):
    """Per-trait Ridge stacking on validation predictions."""
    n_models = len(preds_val_list)
    stacked_test = np.zeros_like(Y_test)
    stacked_val = np.zeros_like(Y_val)

    for t in range(n_traits):
        mv = ~np.isnan(Y_val[:, t])
        if mv.sum() < 10:
            # fallback: simple average
            for k in range(n_models):
                stacked_test[:, t] += preds_test_list[k][:, t] / n_models
                stacked_val[:, t] += preds_val_list[k][:, t] / n_models
            continue
        Xv = np.column_stack([p[mv, t] for p in preds_val_list])
        yv = Y_val[mv, t]
        Xt = np.column_stack([p[:, t] for p in preds_test_list])
        Xv_all = np.column_stack([p[:, t] for p in preds_val_list])
        best_pcc, best_alpha = -999, 1.0
        for alpha in [0.01, 0.1, 1.0, 10.0]:
            reg = SkRidge(alpha=alpha).fit(Xv, yv)
            pred = reg.predict(Xv)
            if np.std(pred) > 1e-8:
                pcc, _ = pearsonr(yv, pred)
                if pcc > best_pcc:
                    best_pcc = pcc; best_alpha = alpha
        reg = SkRidge(alpha=best_alpha).fit(Xv, yv)
        stacked_test[:, t] = reg.predict(Xt)
        stacked_val[:, t] = reg.predict(Xv_all)
    return stacked_val, stacked_test


# ============================================================
# Grid ensemble weight search (3-way)
# ============================================================

def search_3way_weights(pv1, pv2, pv3, Y_val, step=0.1):
    """Search best weights for w1*pv1 + w2*pv2 + w3*pv3, w1+w2+w3=1."""
    best_w, best_pcc = (1/3, 1/3, 1/3), -999
    for w1 in np.arange(0, 1+step/2, step):
        for w2 in np.arange(0, 1-w1+step/2, step):
            w3 = 1 - w1 - w2
            if w3 < -0.01: continue
            w3 = max(w3, 0)
            ens = w1*pv1 + w2*pv2 + w3*pv3
            m = full_metrics(ens, Y_val)
            if m['pcc'] > best_pcc:
                best_pcc = m['pcc']
                best_w = (round(w1,2), round(w2,2), round(w3,2))
    return best_w, best_pcc


# ============================================================
# Main benchmark per dataset
# ============================================================

def run_dataset(ds_name, data_dir, device):
    print(f"\n{'#'*80}")
    print(f"# {ds_name}")
    print(f"{'#'*80}")

    snp_data, phenotype, meta, splits = load_dataset(data_dir)
    n_snps, n_traits, n_samples = snp_data.shape[1], phenotype.shape[1], snp_data.shape[0]
    snp_np, pheno_np = snp_data.numpy(), phenotype.numpy()
    train_idx, val_idx, test_idx = splits['train'], splits['val'], splits['test']
    Y_val, Y_test = pheno_np[val_idx], pheno_np[test_idx]

    print(f"  Samples={n_samples}  SNPs={n_snps}  Traits={n_traits}")
    print(f"  Train/Val/Test = {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    results = {}  # method_name -> {metrics..., n_params, preds_val, preds_test}

    # ---- 1. GBLUP ----
    print(f"\n  >>> GBLUP")
    t0 = time.time()
    gblup_pv, gblup_pt = run_gblup(snp_np, pheno_np, splits)
    tm = time.time() - t0
    results['GBLUP'] = {**full_metrics(gblup_pt, Y_test), 'n_params': 'N/A', 'time': tm,
                         'val': full_metrics(gblup_pv, Y_val)}
    print(f"      PCC={results['GBLUP']['pcc']:.4f}  MSE={results['GBLUP']['mse']:.4f}  "
          f"MAE={results['GBLUP']['mae']:.4f}  ({tm:.1f}s)")

    # ---- 2. MLP ----
    print(f"\n  >>> MLP")
    t0 = time.time()
    mlp, mlp_pv, mlp_pt, mlp_np_ = train_dl(
        MLPBaseline(n_snps*3, n_traits, 512, 0.5),
        snp_data, phenotype, splits, device, n_epochs=60, lr=0.001, wd=1e-3, name="MLP")
    tm = time.time() - t0
    results['MLP'] = {**full_metrics(mlp_pt, Y_test), 'n_params': mlp_np_, 'time': tm,
                       'val': full_metrics(mlp_pv, Y_val)}
    print(f"      PCC={results['MLP']['pcc']:.4f}  MSE={results['MLP']['mse']:.4f}  "
          f"MAE={results['MLP']['mae']:.4f}  ({tm:.1f}s)")

    # ---- 3. DNNGP ----
    print(f"\n  >>> DNNGP")
    t0 = time.time()
    dnngp, dnngp_pv, dnngp_pt, dnngp_np_ = train_dl(
        DNNGP(n_snps, n_traits, 0.3),
        snp_data, phenotype, splits, device, n_epochs=60, lr=0.001, wd=1e-4, name="DNNGP")
    tm = time.time() - t0
    results['DNNGP'] = {**full_metrics(dnngp_pt, Y_test), 'n_params': dnngp_np_, 'time': tm,
                         'val': full_metrics(dnngp_pv, Y_val)}
    print(f"      PCC={results['DNNGP']['pcc']:.4f}  MSE={results['DNNGP']['mse']:.4f}  "
          f"MAE={results['DNNGP']['mae']:.4f}  ({tm:.1f}s)")

    # ---- 4. CNN ----
    print(f"\n  >>> CNN")
    t0 = time.time()
    cnn, cnn_pv, cnn_pt, cnn_np_ = train_dl(
        CNNBaseline(n_snps, n_traits, 0.3),
        snp_data, phenotype, splits, device, n_epochs=60, lr=0.001, wd=1e-4, name="CNN")
    tm = time.time() - t0
    results['CNN'] = {**full_metrics(cnn_pt, Y_test), 'n_params': cnn_np_, 'time': tm,
                       'val': full_metrics(cnn_pv, Y_val)}
    print(f"      PCC={results['CNN']['pcc']:.4f}  MSE={results['CNN']['mse']:.4f}  "
          f"MAE={results['CNN']['mae']:.4f}  ({tm:.1f}s)")

    # ---- 5. Ours V1 (multi-scale CNN) ----
    print(f"\n  >>> Ours V1 (Multi-scale CNN)")
    t0 = time.time()
    v1, v1_pv, v1_pt, v1_np_ = train_dl(
        OurV1(n_snps, n_traits, 256, 0.3),
        snp_data, phenotype, splits, device, n_epochs=60, lr=0.001, wd=1e-4, name="V1")
    tm = time.time() - t0
    results['Ours_V1'] = {**full_metrics(v1_pt, Y_test), 'n_params': v1_np_, 'time': tm,
                           'val': full_metrics(v1_pv, Y_val)}
    print(f"      PCC={results['Ours_V1']['pcc']:.4f}  MSE={results['Ours_V1']['mse']:.4f}  "
          f"MAE={results['Ours_V1']['mae']:.4f}  ({tm:.1f}s)")

    # ---- 6. Ours V1-Deep (4-scale CNN, deeper) ----
    print(f"\n  >>> Ours V1-Deep (4-scale CNN, deeper)")
    t0 = time.time()
    v1d, v1d_pv, v1d_pt, v1d_np_ = train_dl(
        OurV1Deep(n_snps, n_traits, 256, 0.3),
        snp_data, phenotype, splits, device, n_epochs=60, lr=0.001, wd=1e-4, name="V1-Deep")
    tm = time.time() - t0
    results['Ours_V1Deep'] = {**full_metrics(v1d_pt, Y_test), 'n_params': v1d_np_, 'time': tm,
                               'val': full_metrics(v1d_pv, Y_val)}
    print(f"      PCC={results['Ours_V1Deep']['pcc']:.4f}  MSE={results['Ours_V1Deep']['mse']:.4f}  "
          f"MAE={results['Ours_V1Deep']['mae']:.4f}  ({tm:.1f}s)")

    # ---- 7. Ours V2 (anchor-200) ----
    n_anc_small = min(200, n_samples // 5)
    print(f"\n  >>> Ours V2 (Anchor-{n_anc_small})")
    t0 = time.time()
    v2, v2_pv, v2_pt, v2_np_ = train_dl(
        OurV2(n_snps, n_traits, n_anc_small, 256, 0.3),
        snp_data, phenotype, splits, device, n_epochs=60, lr=0.001, wd=1e-4, name="V2")
    tm = time.time() - t0
    results['Ours_V2'] = {**full_metrics(v2_pt, Y_test), 'n_params': v2_np_, 'time': tm,
                           'val': full_metrics(v2_pv, Y_val)}
    print(f"      PCC={results['Ours_V2']['pcc']:.4f}  MSE={results['Ours_V2']['mse']:.4f}  "
          f"MAE={results['Ours_V2']['mae']:.4f}  ({tm:.1f}s)")

    # ---- 8. Ours V2-Large (anchor-400, deeper) ----
    n_anc_large = min(400, n_samples // 3)
    print(f"\n  >>> Ours V2-Large (Anchor-{n_anc_large})")
    t0 = time.time()
    v2l, v2l_pv, v2l_pt, v2l_np_ = train_dl(
        OurV2Large(n_snps, n_traits, n_anc_large, 512, 0.3),
        snp_data, phenotype, splits, device, n_epochs=60, lr=0.001, wd=1e-4, name="V2-Lg")
    tm = time.time() - t0
    results['Ours_V2Large'] = {**full_metrics(v2l_pt, Y_test), 'n_params': v2l_np_, 'time': tm,
                                'val': full_metrics(v2l_pv, Y_val)}
    print(f"      PCC={results['Ours_V2Large']['pcc']:.4f}  MSE={results['Ours_V2Large']['mse']:.4f}  "
          f"MAE={results['Ours_V2Large']['mae']:.4f}  ({tm:.1f}s)")

    # ---- 9. Ensemble V1+V2 (2-way) ----
    print(f"\n  >>> Ensemble V1+V2 (2-way)")
    best_w2, _ = -1, -999
    for w in np.arange(0, 1.05, 0.05):
        e = w*v2_pv + (1-w)*v1_pv
        m = full_metrics(e, Y_val)
        if m['pcc'] > _: _ = m['pcc']; best_w2 = w
    ens2_pt = best_w2*v2_pt + (1-best_w2)*v1_pt
    ens2_pv = best_w2*v2_pv + (1-best_w2)*v1_pv
    results['Ens_V1V2'] = {**full_metrics(ens2_pt, Y_test), 'n_params': 'V1+V2',
                            'val': full_metrics(ens2_pv, Y_val),
                            'weights': f'V2*{best_w2:.2f}+V1*{1-best_w2:.2f}'}
    print(f"      w=(V2*{best_w2:.2f}+V1*{1-best_w2:.2f})  "
          f"PCC={results['Ens_V1V2']['pcc']:.4f}  MSE={results['Ens_V1V2']['mse']:.4f}  "
          f"MAE={results['Ens_V1V2']['mae']:.4f}")

    # ---- 10. 3-way ensemble V1+V2+GBLUP ----
    print(f"\n  >>> 3-way Ensemble V1+V2+GBLUP")
    (w1,w2,w3), _ = search_3way_weights(v1_pv, v2_pv, gblup_pv, Y_val, step=0.05)
    ens3_pt = w1*v1_pt + w2*v2_pt + w3*gblup_pt
    ens3_pv = w1*v1_pv + w2*v2_pv + w3*gblup_pv
    results['Ens_V1V2GBLUP'] = {**full_metrics(ens3_pt, Y_test), 'n_params': 'V1+V2+GBLUP',
                                  'val': full_metrics(ens3_pv, Y_val),
                                  'weights': f'V1*{w1:.2f}+V2*{w2:.2f}+GBLUP*{w3:.2f}'}
    print(f"      w=(V1*{w1:.2f}+V2*{w2:.2f}+GBLUP*{w3:.2f})  "
          f"PCC={results['Ens_V1V2GBLUP']['pcc']:.4f}  MSE={results['Ens_V1V2GBLUP']['mse']:.4f}  "
          f"MAE={results['Ens_V1V2GBLUP']['mae']:.4f}")

    # ---- 11. Stacking ensemble (all DL + GBLUP) ----
    print(f"\n  >>> Stacking Ensemble (Ridge meta-learner on V1+V1D+V2+V2L+GBLUP)")
    all_pv = [v1_pv, v1d_pv, v2_pv, v2l_pv, gblup_pv]
    all_pt = [v1_pt, v1d_pt, v2_pt, v2l_pt, gblup_pt]
    stack_pv, stack_pt = stacking_ensemble(all_pv, all_pt, Y_val, Y_test, n_traits)
    results['Stacking'] = {**full_metrics(stack_pt, Y_test), 'n_params': 'meta',
                            'val': full_metrics(stack_pv, Y_val)}
    print(f"      PCC={results['Stacking']['pcc']:.4f}  MSE={results['Stacking']['mse']:.4f}  "
          f"MAE={results['Stacking']['mae']:.4f}")

    # ---- 12. 3-way best: V1Deep + V2Large + GBLUP ----
    print(f"\n  >>> 3-way Best: V1Deep+V2Large+GBLUP")
    (w1b,w2b,w3b), _ = search_3way_weights(v1d_pv, v2l_pv, gblup_pv, Y_val, step=0.05)
    ens_best_pt = w1b*v1d_pt + w2b*v2l_pt + w3b*gblup_pt
    ens_best_pv = w1b*v1d_pv + w2b*v2l_pv + w3b*gblup_pv
    results['Ens_Best3'] = {**full_metrics(ens_best_pt, Y_test), 'n_params': 'V1D+V2L+GBLUP',
                             'val': full_metrics(ens_best_pv, Y_val),
                             'weights': f'V1D*{w1b:.2f}+V2L*{w2b:.2f}+GBLUP*{w3b:.2f}'}
    print(f"      w=(V1D*{w1b:.2f}+V2L*{w2b:.2f}+GBLUP*{w3b:.2f})  "
          f"PCC={results['Ens_Best3']['pcc']:.4f}  MSE={results['Ens_Best3']['mse']:.4f}  "
          f"MAE={results['Ens_Best3']['mae']:.4f}")

    return results, Y_test, Y_val


def print_table(all_results, datasets):
    """Print a pretty results table."""
    methods_order = ['GBLUP', 'MLP', 'DNNGP', 'CNN',
                     'Ours_V1', 'Ours_V1Deep', 'Ours_V2', 'Ours_V2Large',
                     'Ens_V1V2', 'Ens_V1V2GBLUP', 'Ens_Best3', 'Stacking']

    for ds in datasets:
        res = all_results[ds]
        print(f"\n{'='*100}")
        print(f"  {ds}")
        print(f"{'='*100}")
        print(f"  {'Method':<20} {'Test PCC':>9} {'Spearman':>9} {'MSE':>9} {'MAE':>9}"
              f" | {'Val PCC':>9} {'Val MSE':>9} {'Val MAE':>9} | {'Params':>12}")
        print(f"  {'-'*95}")
        for m in methods_order:
            if m not in res: continue
            r = res[m]
            vr = r.get('val', {})
            np_ = r.get('n_params', '—')
            ps = f"{np_:,}" if isinstance(np_, int) else str(np_)
            print(f"  {m:<20} {r['pcc']:>9.4f} {r['spearman']:>9.4f} {r['mse']:>9.4f} {r['mae']:>9.4f}"
                  f" | {vr.get('pcc',0):>9.4f} {vr.get('mse',0):>9.4f} {vr.get('mae',0):>9.4f} | {ps:>12}")

        # Ranking
        ranked = sorted([(m, res[m]) for m in methods_order if m in res],
                        key=lambda x: x[1]['pcc'], reverse=True)
        print(f"\n  Ranking by Test PCC:")
        for i, (m, r) in enumerate(ranked):
            tag = " 🏆" if i == 0 else ""
            print(f"    {i+1:2d}. {m:<20} PCC={r['pcc']:.4f}  MSE={r['mse']:.4f}  MAE={r['mae']:.4f}{tag}")


def main():
    print("=" * 100)
    print("ENHANCED BENCHMARK — Improvement Strategies + Full Metrics (PCC/Spearman/MSE/MAE)")
    print("=" * 100)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    all_results = {}
    datasets_info = [
        ("GSTP007", "data/processed/GSTP007_full_10000snps_processed"),
        ("Rice469", "data/processed/rice469"),
    ]

    for ds_name, ds_path in datasets_info:
        res, _, _ = run_dataset(ds_name, ds_path, device)
        all_results[ds_name] = res

    # ============ Final tables ============
    print(f"\n\n{'#'*100}")
    print("FINAL COMPREHENSIVE RESULTS")
    print(f"{'#'*100}")
    print_table(all_results, [d[0] for d in datasets_info])

    # ============ Cross-dataset summary ============
    print(f"\n{'='*100}")
    print("CROSS-DATASET SUMMARY (Test PCC / MSE / MAE)")
    print(f"{'='*100}")
    methods_order = ['GBLUP', 'MLP', 'DNNGP', 'CNN',
                     'Ours_V1', 'Ours_V1Deep', 'Ours_V2', 'Ours_V2Large',
                     'Ens_V1V2', 'Ens_V1V2GBLUP', 'Ens_Best3', 'Stacking']
    ds_names = [d[0] for d in datasets_info]
    header = f"  {'Method':<20}"
    for ds in ds_names:
        header += f" | {ds+' PCC':>10} {ds+' MSE':>10} {ds+' MAE':>10}"
    print(header)
    print(f"  {'-'*len(header)}")
    for m in methods_order:
        line = f"  {m:<20}"
        for ds in ds_names:
            r = all_results.get(ds, {}).get(m, {})
            line += f" | {r.get('pcc',0):>10.4f} {r.get('mse',0):>10.4f} {r.get('mae',0):>10.4f}"
        print(line)

    # ============ Save ============
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        return obj

    out = Path("data/processed/GSTP007_full_10000snps_processed/enhanced_benchmark_results.json")
    with open(out, 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"\n✓ Results saved to {out}")

    return 0


if __name__ == '__main__':
    exit(main())

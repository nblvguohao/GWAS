#!/usr/bin/env python
"""
Benchmark: SNP scaling (10K vs 50K) + multi-trait selection + best encoding/loss combos.
Tests the hypothesis that more SNPs and better selection improve performance.
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
# Metrics & Loss
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
            'mse': float(np.mean(mses)), 'mae': float(np.mean(maes))}

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
# Models
# ============================================================

class MultiScaleCNN(nn.Module):
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
        if x.dim() == 2: x = x.unsqueeze(1)
        else: x = x.transpose(1, 2)
        x = self.bn1(torch.cat([F.relu(self.conv_k3(x)), F.relu(self.conv_k7(x)), F.relu(self.conv_k15(x))], 1))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(self.pool(x).view(x.size(0), -1))


class MLPBaseline(nn.Module):
    def __init__(self, input_dim, n_traits, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_traits),
        )
    def forward(self, x):
        if x.dim() == 3: x = x.reshape(x.size(0), -1)
        return self.net(x)


# ============================================================
# Training
# ============================================================

def predict_batch(model, data, indices, device, bs=64):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(indices), bs):
            bi = indices[i:i+bs]
            preds.append(model(data[bi].to(device)).cpu())
    return torch.cat(preds).numpy()


def train_model(model, train_data, phenotype, splits, device,
                n_epochs=80, lr=0.001, wd=1e-4, bs=32, name="Model",
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
    return pv, pt, n_params


def run_gblup(X_additive, pheno_np, splits, lambdas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]):
    train_idx, val_idx, test_idx = [np.array(splits[k]) for k in ('train','val','test')]
    X = X_additive.astype(np.float64)
    X_c = X - X.mean(axis=0)
    K = X_c @ X_c.T / X_c.shape[1]
    K_tr = K[np.ix_(train_idx, train_idx)]
    K_va = K[np.ix_(val_idx, train_idx)]
    K_te = K[np.ix_(test_idx, train_idx)]
    Y_tr = pheno_np[train_idx]
    n_tr, n_traits = len(train_idx), pheno_np.shape[1]
    pv, pt = np.zeros((len(val_idx), n_traits)), np.zeros((len(test_idx), n_traits))
    for t in range(n_traits):
        y = Y_tr[:, t].copy(); y[np.isnan(y)] = 0.0
        best_pcc = -999
        for lam in lambdas:
            a = np.linalg.solve(K_tr + lam * np.eye(n_tr), y)
            pred_v = K_va @ a
            mv = ~np.isnan(pheno_np[val_idx, t])
            pcc = pearsonr(pheno_np[val_idx[mv], t], pred_v[mv])[0] if mv.sum() > 5 and np.std(pred_v[mv]) > 1e-8 else 0.0
            if pcc > best_pcc:
                best_pcc = pcc; pv[:, t] = pred_v; pt[:, t] = K_te @ a
    return pv, pt


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
        best_a, best_pcc = 1.0, -999
        for a in [0.01, 0.1, 1.0, 10.0]:
            r = Ridge(alpha=a).fit(Xv, yv)
            pred = r.predict(Xv)
            if np.std(pred) > 1e-8:
                pcc = pearsonr(yv, pred)[0]
                if pcc > best_pcc: best_pcc = pcc; best_a = a
        r = Ridge(alpha=best_a).fit(Xv, yv)
        st[:, t] = r.predict(Xt); sv[:, t] = r.predict(Xv_all)
    return sv, st


def onehot_to_additive(g):
    return g[:, :, 1] + 2 * g[:, :, 2]


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 100)
    print("SNP SCALING + MULTI-TRAIT SELECTION BENCHMARK")
    print("=" * 100)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_dir = Path("data/processed/GSTP007_full_10000snps_processed")
    pheno_np = np.load(data_dir / "phenotype_scaled.npy")
    with open(data_dir / "split.json") as f:
        splits = json.load(f)
    n_traits = pheno_np.shape[1]
    val_idx, test_idx = np.array(splits['val']), np.array(splits['test'])
    Y_val, Y_test = pheno_np[val_idx], pheno_np[test_idx]
    phenotype_t = torch.from_numpy(pheno_np).float()

    # Load different data configurations
    configs = {}

    # 1. Original 10K one-hot → additive
    g_oh = np.load(data_dir / "genotype_onehot.npy")
    configs['10K_single_trait'] = onehot_to_additive(g_oh)
    del g_oh

    # 2. Multi-trait selected 10K
    configs['10K_multi_trait'] = np.load(data_dir / "genotype_10k_mt_additive.npy")

    # 3. 50K multi-trait selected
    configs['50K_multi_trait'] = np.load(data_dir / "genotype_50k_additive.npy")

    for name, data in configs.items():
        print(f"\n  {name}: shape={data.shape}")

    results = {}

    for cfg_name, geno_add in configs.items():
        print(f"\n{'#'*80}")
        print(f"# CONFIG: {cfg_name} — {geno_add.shape[1]} SNPs")
        print(f"{'#'*80}")

        n_snps = geno_add.shape[1]
        X_t = torch.from_numpy(geno_add).float()
        all_pv, all_pt, all_names = [], [], []

        # GBLUP
        print(f"\n  >>> GBLUP ({cfg_name})")
        t0 = time.time()
        gpv, gpt = run_gblup(geno_add, pheno_np, splits)
        tm = time.time() - t0
        m = full_metrics(gpt, Y_test)
        print(f"      PCC={m['pcc']:.4f}  MSE={m['mse']:.4f}  MAE={m['mae']:.4f}  ({tm:.0f}s)")
        results[f'{cfg_name}_GBLUP'] = m
        all_pv.append(gpv); all_pt.append(gpt); all_names.append('GBLUP')

        # MLP
        print(f"\n  >>> MLP ({cfg_name})")
        t0 = time.time()
        pv, pt, np_ = train_model(MLPBaseline(n_snps, n_traits), X_t, phenotype_t, splits, device,
                                   name=f"MLP-{cfg_name}", n_epochs=60, lr=0.001, bs=32)
        tm = time.time() - t0
        m = full_metrics(pt, Y_test)
        print(f"      PCC={m['pcc']:.4f}  MSE={m['mse']:.4f}  MAE={m['mae']:.4f}  ({tm:.0f}s)")
        results[f'{cfg_name}_MLP'] = m
        all_pv.append(pv); all_pt.append(pt); all_names.append('MLP')

        # CNN-Additive (MSE loss)
        print(f"\n  >>> CNN-Add ({cfg_name})")
        t0 = time.time()
        pv, pt, np_ = train_model(MultiScaleCNN(n_snps, n_traits, 1), X_t, phenotype_t, splits, device,
                                   name=f"CNN-{cfg_name}", n_epochs=80, lr=0.001, bs=32)
        tm = time.time() - t0
        m = full_metrics(pt, Y_test)
        print(f"      PCC={m['pcc']:.4f}  MSE={m['mse']:.4f}  MAE={m['mae']:.4f}  ({tm:.0f}s)")
        results[f'{cfg_name}_CNN'] = m
        all_pv.append(pv); all_pt.append(pt); all_names.append('CNN')

        # CNN-Additive + Huber
        print(f"\n  >>> CNN-Add-Huber ({cfg_name})")
        t0 = time.time()
        pv, pt, np_ = train_model(MultiScaleCNN(n_snps, n_traits, 1), X_t, phenotype_t, splits, device,
                                   name=f"Huber-{cfg_name}", n_epochs=80, lr=0.001, bs=32, loss_fn='huber')
        tm = time.time() - t0
        m = full_metrics(pt, Y_test)
        print(f"      PCC={m['pcc']:.4f}  MSE={m['mse']:.4f}  MAE={m['mae']:.4f}  ({tm:.0f}s)")
        results[f'{cfg_name}_CNN_Huber'] = m
        all_pv.append(pv); all_pt.append(pt); all_names.append('CNN_Huber')

        # Stacking of all 4
        print(f"\n  >>> Stacking ({cfg_name})")
        spv, spt = stacking(all_pv, all_pt, Y_val, Y_test, n_traits)
        m = full_metrics(spt, Y_test)
        print(f"      PCC={m['pcc']:.4f}  MSE={m['mse']:.4f}  MAE={m['mae']:.4f}")
        results[f'{cfg_name}_Stacking'] = m

        del X_t
        torch.cuda.empty_cache()

    # ============ Final comparison ============
    print(f"\n\n{'='*100}")
    print("FINAL COMPARISON: SNP Scaling × Selection × Method")
    print(f"{'='*100}")
    print(f"  {'Configuration':<35} {'PCC':>8} {'Spearman':>8} {'MSE':>8} {'MAE':>8}")
    print(f"  {'-'*75}")
    ranked = sorted(results.items(), key=lambda x: x[1]['pcc'], reverse=True)
    for i, (name, m) in enumerate(ranked):
        tag = " 🏆" if i == 0 else ""
        print(f"  {name:<35} {m['pcc']:>8.4f} {m['spearman']:>8.4f} {m['mse']:>8.4f} {m['mae']:>8.4f}{tag}")

    # Grouped comparison
    print(f"\n\nGROUPED: Same method across SNP configs")
    print(f"{'='*100}")
    for method in ['GBLUP', 'MLP', 'CNN', 'CNN_Huber', 'Stacking']:
        print(f"\n  {method}:")
        for cfg in ['10K_single_trait', '10K_multi_trait', '50K_multi_trait']:
            key = f'{cfg}_{method}'
            if key in results:
                m = results[key]
                print(f"    {cfg:<25} PCC={m['pcc']:.4f}  MSE={m['mse']:.4f}  MAE={m['mae']:.4f}")

    # Save results
    def ser(obj):
        if isinstance(obj, dict): return {k: ser(v) for k, v in obj.items()}
        if isinstance(obj, list): return [ser(v) for v in obj]
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        return obj

    out = data_dir / "snp_scaling_results.json"
    with open(out, 'w') as f:
        json.dump(ser(results), f, indent=2)
    print(f"\n✓ Results saved to {out}")


if __name__ == '__main__':
    main()

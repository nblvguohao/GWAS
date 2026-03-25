#!/usr/bin/env python
"""
Full benchmark: 
  1. Analyze further improvement potential
  2. Fairly reproduce baselines (GBLUP, DNNGP, MLP, CNN, our V1/V2/Ensemble)
  3. Run on both GSTP007 and Rice469 datasets
All methods use identical train/val/test splits and NaN-safe evaluation.
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
from tqdm import tqdm
import time

# ============================================================
# Utility
# ============================================================

def nan_safe_mse(pred, target):
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return ((pred - target) ** 2 * mask).sum() / mask.sum()


def eval_pcc(preds, targets):
    """Per-trait Pearson & Spearman, NaN-safe."""
    pccs, sccs = [], []
    for t in range(targets.shape[1]):
        m = ~np.isnan(targets[:, t])
        if m.sum() > 10 and np.std(preds[m, t]) > 1e-8:
            p, _ = pearsonr(targets[m, t], preds[m, t])
            s, _ = spearmanr(targets[m, t], preds[m, t])
            pccs.append(float(p)); sccs.append(float(s))
        else:
            pccs.append(0.0); sccs.append(0.0)
    return pccs, sccs


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
# Baseline 1: GBLUP (Ridge regression on GRM)
# ============================================================

def run_gblup(snp_np, pheno_np, splits, lambdas=[0.1, 1.0, 10.0, 100.0]):
    """GBLUP via kernel ridge regression on GRM. Fast: O(n^3) not O(p^3)."""
    n_samples, n_snps, _ = snp_np.shape
    X = snp_np.reshape(n_samples, -1).astype(np.float64)
    
    train_idx = np.array(splits['train'])
    val_idx = np.array(splits['val'])
    test_idx = np.array(splits['test'])
    
    # Compute GRM (genomic relationship matrix) K = X X^T / p
    # This is n×n (e.g. 1495×1495) — much smaller than p×p
    print("    Computing GRM...", end=" ", flush=True)
    X_centered = X - X.mean(axis=0)
    K = X_centered @ X_centered.T / X_centered.shape[1]
    print("done.")
    
    K_train = K[np.ix_(train_idx, train_idx)]
    K_val_train = K[np.ix_(val_idx, train_idx)]
    K_test_train = K[np.ix_(test_idx, train_idx)]
    
    Y_train = pheno_np[train_idx]
    Y_val = pheno_np[val_idx]
    Y_test = pheno_np[test_idx]
    n_traits = Y_train.shape[1]
    n_tr = len(train_idx)
    
    best_preds_val = np.zeros_like(Y_val)
    best_preds_test = np.zeros_like(Y_test)
    
    for t in range(n_traits):
        mask_tr = ~np.isnan(Y_train[:, t])
        mask_val = ~np.isnan(Y_val[:, t])
        if mask_tr.sum() < 10:
            continue
        
        # For simplicity use all training samples (fill NaN with 0)
        y_tr = Y_train[:, t].copy()
        y_tr[np.isnan(y_tr)] = 0.0
        
        best_val_pcc = -999
        for lam in lambdas:
            # Solve (K_train + λI) α = y_train
            alpha = np.linalg.solve(K_train + lam * np.eye(n_tr), y_tr)
            pred_val = K_val_train @ alpha
            
            if mask_val.sum() > 5 and np.std(pred_val[mask_val]) > 1e-8:
                pcc, _ = pearsonr(Y_val[mask_val, t], pred_val[mask_val])
            else:
                pcc = 0.0
            
            if pcc > best_val_pcc:
                best_val_pcc = pcc
                best_preds_val[:, t] = pred_val
                best_preds_test[:, t] = K_test_train @ alpha
    
    test_pccs, test_sccs = eval_pcc(best_preds_test, Y_test)
    val_pccs, _ = eval_pcc(best_preds_val, Y_val)
    return {
        'test_pcc': float(np.mean(test_pccs)),
        'val_pcc': float(np.mean(val_pccs)),
        'test_pccs': test_pccs,
        'test_spearman': float(np.mean(test_sccs)),
    }


# ============================================================
# Baseline 2: MLP
# ============================================================

class MLPBaseline(nn.Module):
    def __init__(self, input_dim, n_traits, hidden=512, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 4),
            nn.BatchNorm1d(hidden // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 4, n_traits),
        )
    def forward(self, x):
        return self.net(x.reshape(x.size(0), -1))


# ============================================================
# Baseline 3: DNNGP (Deep Neural Network for Genomic Prediction)
# Ma et al. 2022 — 1D-CNN + Transformer-like extraction
# ============================================================

class DNNGP(nn.Module):
    """Simplified DNNGP: Conv1D feature extraction + dense layers."""
    def __init__(self, n_snps, n_traits, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 32, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_traits),
        )
    def forward(self, x):
        x = x.transpose(1, 2)  # (B, 3, SNPs)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


# ============================================================
# Baseline 4: Simple CNN baseline
# ============================================================

class CNNBaseline(nn.Module):
    def __init__(self, n_snps, n_traits, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.AdaptiveAvgPool1d(16),
        )
        self.head = nn.Sequential(
            nn.Linear(128 * 16, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, n_traits),
        )
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


# ============================================================
# Our models (V1 & V2)
# ============================================================

class OurV1_MultiScaleCNN(nn.Module):
    def __init__(self, n_snps, n_traits, d_hidden=256, dropout=0.3):
        super().__init__()
        self.conv_k3 = nn.Conv1d(3, 32, kernel_size=3, padding=1)
        self.conv_k7 = nn.Conv1d(3, 32, kernel_size=7, padding=3)
        self.conv_k15 = nn.Conv1d(3, 32, kernel_size=15, padding=7)
        self.bn1 = nn.BatchNorm1d(96)
        self.conv2 = nn.Conv1d(96, 128, kernel_size=5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=5, padding=2, stride=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(64)
        self.head = nn.Sequential(
            nn.Linear(128 * 64, d_hidden), nn.BatchNorm1d(d_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2), nn.BatchNorm1d(d_hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, n_traits),
        )
    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.cat([F.relu(self.conv_k3(x)), F.relu(self.conv_k7(x)), F.relu(self.conv_k15(x))], dim=1)
        x = self.bn1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).view(x.size(0), -1)
        return self.head(x)


class OurV2_AnchorSimilarity(nn.Module):
    def __init__(self, n_snps, n_traits, n_anchors=200, d_hidden=256, dropout=0.3):
        super().__init__()
        self.anchors = nn.Parameter(torch.randn(n_anchors, n_snps * 3) * 0.01)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.head = nn.Sequential(
            nn.Linear(n_anchors, d_hidden), nn.BatchNorm1d(d_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2), nn.BatchNorm1d(d_hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, n_traits),
        )
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x_n = F.normalize(x, dim=1)
        a_n = F.normalize(self.anchors, dim=1)
        sim = torch.matmul(x_n, a_n.t()) / self.temperature.abs().clamp(min=0.1)
        return self.head(sim)


# ============================================================
# Generic DL trainer
# ============================================================

def train_dl_model(model, snp_data, phenotype, splits, device,
                   n_epochs=50, lr=0.001, wd=1e-4, bs=32, name="Model"):
    model = model.to(device)
    opt = Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    
    train_idx, val_idx, test_idx = splits['train'], splits['val'], splits['test']
    best_val, best_state, patience_ctr = -999, None, 0
    
    for epoch in range(1, n_epochs + 1):
        model.train()
        idx = train_idx.copy(); np.random.shuffle(idx)
        tot_loss, nb = 0, 0
        for i in range(0, len(idx), bs):
            bi = idx[i:i+bs]
            x = snp_data[bi].to(device)
            y = phenotype[bi].to(device)
            opt.zero_grad()
            loss = nan_safe_mse(model(x), y)
            if torch.isnan(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += loss.item(); nb += 1
        sched.step()
        if nb == 0: continue
        
        # Validate
        model.eval()
        with torch.no_grad():
            vp = []; 
            for i in range(0, len(val_idx), 64):
                bi = val_idx[i:i+64]
                vp.append(model(snp_data[bi].to(device)).cpu())
            vp = torch.cat(vp).numpy()
        val_pccs, _ = eval_pcc(vp, phenotype[val_idx].numpy())
        vpcc = np.mean(val_pccs)
        
        if vpcc > best_val:
            best_val = vpcc; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= 12: break
        
        if epoch % 10 == 0:
            print(f"    [{name}] ep {epoch:3d}  loss={tot_loss/nb:.4f}  val_PCC={vpcc:.4f}")
    
    if best_state: model.load_state_dict(best_state)
    model = model.to(device).eval()
    with torch.no_grad():
        tp = []
        for i in range(0, len(test_idx), 64):
            bi = test_idx[i:i+64]
            tp.append(model(snp_data[bi].to(device)).cpu())
        tp = torch.cat(tp).numpy()
    
    test_pccs, test_sccs = eval_pcc(tp, phenotype[test_idx].numpy())
    return {
        'test_pcc': float(np.mean(test_pccs)),
        'val_pcc': float(best_val),
        'test_pccs': test_pccs,
        'test_spearman': float(np.mean(test_sccs)),
        'n_params': sum(p.numel() for p in model.parameters()),
    }, tp


# ============================================================
# Main
# ============================================================

def run_benchmark_on_dataset(dataset_name, data_dir, device):
    print(f"\n{'#'*80}")
    print(f"# BENCHMARK: {dataset_name}")
    print(f"{'#'*80}")
    
    snp_data, phenotype, meta, splits = load_dataset(data_dir)
    n_snps = snp_data.shape[1]
    n_traits = phenotype.shape[1]
    n_samples = snp_data.shape[0]
    
    print(f"  Samples: {n_samples}, SNPs: {n_snps}, Traits: {n_traits}")
    print(f"  Train/Val/Test: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    
    snp_np = snp_data.numpy()
    pheno_np = phenotype.numpy()
    
    results = {}
    
    # --- GBLUP ---
    print(f"\n  [1/6] GBLUP (Ridge regression)...")
    t0 = time.time()
    results['GBLUP'] = run_gblup(snp_np, pheno_np, splits)
    results['GBLUP']['time'] = time.time() - t0
    results['GBLUP']['n_params'] = 'N/A'
    print(f"    Test PCC: {results['GBLUP']['test_pcc']:.4f}  ({results['GBLUP']['time']:.1f}s)")
    
    # --- MLP ---
    print(f"\n  [2/6] MLP baseline...")
    t0 = time.time()
    mlp = MLPBaseline(n_snps * 3, n_traits, hidden=512, dropout=0.5)
    results['MLP'], _ = train_dl_model(mlp, snp_data, phenotype, splits, device,
                                        n_epochs=50, lr=0.001, wd=1e-3, name="MLP")
    results['MLP']['time'] = time.time() - t0
    print(f"    Test PCC: {results['MLP']['test_pcc']:.4f}  ({results['MLP']['time']:.1f}s)")
    
    # --- DNNGP ---
    print(f"\n  [3/6] DNNGP...")
    t0 = time.time()
    dnngp = DNNGP(n_snps, n_traits, dropout=0.3)
    results['DNNGP'], _ = train_dl_model(dnngp, snp_data, phenotype, splits, device,
                                          n_epochs=50, lr=0.001, wd=1e-4, name="DNNGP")
    results['DNNGP']['time'] = time.time() - t0
    print(f"    Test PCC: {results['DNNGP']['test_pcc']:.4f}  ({results['DNNGP']['time']:.1f}s)")
    
    # --- CNN ---
    print(f"\n  [4/6] CNN baseline...")
    t0 = time.time()
    cnn = CNNBaseline(n_snps, n_traits, dropout=0.3)
    results['CNN'], _ = train_dl_model(cnn, snp_data, phenotype, splits, device,
                                        n_epochs=50, lr=0.001, wd=1e-4, name="CNN")
    results['CNN']['time'] = time.time() - t0
    print(f"    Test PCC: {results['CNN']['test_pcc']:.4f}  ({results['CNN']['time']:.1f}s)")
    
    # --- Ours V1 ---
    print(f"\n  [5/6] Ours V1 (Multi-scale CNN)...")
    t0 = time.time()
    v1 = OurV1_MultiScaleCNN(n_snps, n_traits, d_hidden=256, dropout=0.3)
    results['Ours_V1'], preds_v1 = train_dl_model(v1, snp_data, phenotype, splits, device,
                                                    n_epochs=50, lr=0.001, wd=1e-4, name="Ours_V1")
    results['Ours_V1']['time'] = time.time() - t0
    print(f"    Test PCC: {results['Ours_V1']['test_pcc']:.4f}  ({results['Ours_V1']['time']:.1f}s)")
    
    # --- Ours V2 ---
    print(f"\n  [6/6] Ours V2 (Anchor Similarity)...")
    n_anchors = min(200, n_samples // 5)  # scale anchors to dataset size
    t0 = time.time()
    v2 = OurV2_AnchorSimilarity(n_snps, n_traits, n_anchors=n_anchors, d_hidden=256, dropout=0.3)
    results['Ours_V2'], preds_v2 = train_dl_model(v2, snp_data, phenotype, splits, device,
                                                    n_epochs=50, lr=0.001, wd=1e-4, name="Ours_V2")
    results['Ours_V2']['time'] = time.time() - t0
    print(f"    Test PCC: {results['Ours_V2']['test_pcc']:.4f}  ({results['Ours_V2']['time']:.1f}s)")
    
    # --- Ensemble V1+V2 ---
    test_idx = splits['test']
    targets = pheno_np[test_idx]
    val_idx = splits['val']
    
    # Get val predictions for weight search
    v1.to(device).eval(); v2.to(device).eval()
    with torch.no_grad():
        vp1 = torch.cat([v1(snp_data[val_idx[i:i+64]].to(device)).cpu() for i in range(0, len(val_idx), 64)]).numpy()
        vp2 = torch.cat([v2(snp_data[val_idx[i:i+64]].to(device)).cpu() for i in range(0, len(val_idx), 64)]).numpy()
    val_tgt = pheno_np[val_idx]
    
    best_w, best_ens_val = 0.5, -999
    for w in np.arange(0.0, 1.05, 0.1):
        ens = w * vp2 + (1 - w) * vp1
        pc, _ = eval_pcc(ens, val_tgt)
        avg = np.mean(pc)
        if avg > best_ens_val:
            best_ens_val = avg; best_w = w
    
    ens_test = best_w * preds_v2 + (1 - best_w) * preds_v1
    ens_pccs, ens_sccs = eval_pcc(ens_test, targets)
    results['Ours_Ensemble'] = {
        'test_pcc': float(np.mean(ens_pccs)),
        'val_pcc': float(best_ens_val),
        'test_pccs': ens_pccs,
        'test_spearman': float(np.mean(ens_sccs)),
        'n_params': 'V1+V2',
        'weight': f'V2*{best_w:.1f}+V1*{1-best_w:.1f}',
    }
    print(f"\n  Ensemble (V2*{best_w:.1f}+V1*{1-best_w:.1f}): Test PCC = {results['Ours_Ensemble']['test_pcc']:.4f}")
    
    return results


def main():
    print("=" * 80)
    print("COMPREHENSIVE BENCHMARK — All Methods × All Datasets")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    all_results = {}
    
    # Dataset 1: GSTP007 (10K SNPs, 1495 samples, 32 traits)
    all_results['GSTP007'] = run_benchmark_on_dataset(
        "GSTP007 (1495 samples, 10K SNPs, 32 traits)",
        "data/processed/GSTP007_full_10000snps_processed",
        device
    )
    
    # Dataset 2: Rice469 (469 samples, 500 SNPs, 6 traits)
    all_results['Rice469'] = run_benchmark_on_dataset(
        "Rice469 (469 samples, 500 SNPs, 6 traits)",
        "data/processed/rice469",
        device
    )
    
    # ============================================================
    # Final report
    # ============================================================
    print(f"\n\n{'='*80}")
    print("FINAL COMPREHENSIVE RESULTS")
    print(f"{'='*80}")
    
    methods = ['GBLUP', 'MLP', 'DNNGP', 'CNN', 'Ours_V1', 'Ours_V2', 'Ours_Ensemble']
    datasets = ['GSTP007', 'Rice469']
    
    # Table header
    print(f"\n{'Method':<20}", end="")
    for ds in datasets:
        print(f" | {ds+' PCC':>12} {ds+' Spear':>12}", end="")
    print(f" | {'Avg PCC':>10}")
    print("-" * 90)
    
    for method in methods:
        print(f"{method:<20}", end="")
        avg_pccs = []
        for ds in datasets:
            r = all_results[ds].get(method, {})
            pcc = r.get('test_pcc', 0)
            scc = r.get('test_spearman', 0)
            avg_pccs.append(pcc)
            print(f" | {pcc:>12.4f} {scc:>12.4f}", end="")
        print(f" | {np.mean(avg_pccs):>10.4f}")
    
    # Per-dataset best
    for ds in datasets:
        print(f"\n  {ds} ranking:")
        ranked = sorted(all_results[ds].items(), key=lambda x: x[1].get('test_pcc', 0), reverse=True)
        for i, (name, r) in enumerate(ranked):
            marker = "🏆" if i == 0 else "  "
            print(f"    {marker} {i+1}. {name:<20} PCC={r.get('test_pcc',0):.4f}")
    
    # Improvement analysis
    print(f"\n{'='*80}")
    print("IMPROVEMENT ANALYSIS")
    print(f"{'='*80}")
    
    for ds in datasets:
        gblup_pcc = all_results[ds]['GBLUP']['test_pcc']
        ens_pcc = all_results[ds]['Ours_Ensemble']['test_pcc']
        v2_pcc = all_results[ds]['Ours_V2']['test_pcc']
        best_baseline = max(all_results[ds]['DNNGP']['test_pcc'],
                           all_results[ds]['CNN']['test_pcc'],
                           all_results[ds]['MLP']['test_pcc'])
        
        print(f"\n  {ds}:")
        print(f"    GBLUP:          {gblup_pcc:.4f}")
        print(f"    Best DL baseline: {best_baseline:.4f}")
        print(f"    Ours V2:        {v2_pcc:.4f}  (vs GBLUP: {(v2_pcc-gblup_pcc)/abs(gblup_pcc)*100:+.1f}%)")
        print(f"    Ours Ensemble:  {ens_pcc:.4f}  (vs GBLUP: {(ens_pcc-gblup_pcc)/abs(gblup_pcc)*100:+.1f}%)")
    
    # Save
    serializable = {}
    for ds, ds_results in all_results.items():
        serializable[ds] = {}
        for method, r in ds_results.items():
            serializable[ds][method] = {
                k: (float(v) if isinstance(v, (float, np.floating)) else
                    [float(x) for x in v] if isinstance(v, list) else str(v))
                for k, v in r.items()
            }
    
    out_file = Path("data/processed/GSTP007_full_10000snps_processed/full_benchmark_results.json")
    with open(out_file, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\n✓ All results saved to {out_file}")
    
    return 0


if __name__ == '__main__':
    exit(main())

#!/usr/bin/env python
"""
Train the best model (V2 Anchor Similarity) with full epochs and save results.
Based on diagnosis findings from diagnose_and_fix.py.
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
from tqdm import tqdm


class AnchorSimilarityModel(nn.Module):
    """
    V2: Anchor-similarity model (GBLUP-inspired deep learning).
    
    Computes cosine similarity between each sample's SNP vector and a set of
    learnable anchor points, then predicts traits from the similarity features.
    """
    def __init__(self, n_snps, n_traits, n_anchors=200, d_hidden=256, dropout=0.3):
        super().__init__()
        self.n_snps = n_snps
        self.n_anchors = n_anchors
        input_dim = n_snps * 3
        
        # Learnable anchor points in SNP space
        self.anchors = nn.Parameter(torch.randn(n_anchors, input_dim) * 0.01)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(n_anchors, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.BatchNorm1d(d_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, n_traits)
        )
    
    def forward(self, snp_data, graph_data=None):
        batch_size = snp_data.size(0)
        x = snp_data.reshape(batch_size, -1)
        
        x_norm = F.normalize(x, dim=1)
        a_norm = F.normalize(self.anchors, dim=1)
        similarity = torch.matmul(x_norm, a_norm.t()) / self.temperature.abs().clamp(min=0.1)
        
        return self.head(similarity)


class MultiScaleCNNModel(nn.Module):
    """
    V1: Multi-scale CNN model (stripped of all constant components).
    """
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
            nn.Linear(128 * 64, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.BatchNorm1d(d_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, n_traits)
        )
    
    def forward(self, snp_data, graph_data=None):
        x = snp_data.transpose(1, 2)
        x3 = F.relu(self.conv_k3(x))
        x7 = F.relu(self.conv_k7(x))
        x15 = F.relu(self.conv_k15(x))
        x = torch.cat([x3, x7, x15], dim=1)
        x = self.bn1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


def nan_safe_mse_loss(pred, target):
    """MSE loss that masks NaN values in target."""
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    diff = (pred - target) ** 2
    return (diff * mask).sum() / mask.sum()


def evaluate_full(model, snp_data, phenotype, indices, device):
    """Full evaluation with per-trait PCC and Spearman."""
    model.eval()
    with torch.no_grad():
        all_preds = []
        batch_size = 64
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            x = snp_data[batch_idx].to(device)
            pred = model(x).cpu()
            all_preds.append(pred)
        preds = torch.cat(all_preds, dim=0).numpy()
    
    targets = phenotype[indices].numpy()
    
    pearson_per_trait = []
    spearman_per_trait = []
    for t in range(targets.shape[1]):
        mask = ~np.isnan(targets[:, t])
        if mask.sum() > 10 and np.std(preds[mask, t]) > 1e-8:
            pcc, _ = pearsonr(targets[mask, t], preds[mask, t])
            scc, _ = spearmanr(targets[mask, t], preds[mask, t])
            pearson_per_trait.append(float(pcc))
            spearman_per_trait.append(float(scc))
        else:
            pearson_per_trait.append(0.0)
            spearman_per_trait.append(0.0)
    
    return {
        'pearson_per_trait': pearson_per_trait,
        'spearman_per_trait': spearman_per_trait,
        'pearson_avg': float(np.mean(pearson_per_trait)),
        'spearman_avg': float(np.mean(spearman_per_trait)),
    }


def train_model(model, snp_data, phenotype, splits, device,
                n_epochs=60, lr=0.001, weight_decay=1e-4, batch_size=32,
                model_name="Model"):
    """Train with full logging."""
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Epochs: {n_epochs}, LR: {lr}, WD: {weight_decay}, BS: {batch_size}")
    print(f"{'='*70}")
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    train_idx = splits['train']
    val_idx = splits['val']
    test_idx = splits['test']
    
    best_val_pcc = -999
    best_state = None
    patience = 15
    patience_counter = 0
    history = []
    
    for epoch in range(1, n_epochs + 1):
        model.train()
        indices = train_idx.copy()
        np.random.shuffle(indices)
        total_loss = 0
        n_batches = 0
        
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            x = snp_data[batch_idx].to(device)
            y = phenotype[batch_idx].to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = nan_safe_mse_loss(pred, y)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        if n_batches == 0:
            continue
        
        avg_loss = total_loss / n_batches
        
        # Validate
        val_metrics = evaluate_full(model, snp_data, phenotype, val_idx, device)
        val_pcc = val_metrics['pearson_avg']
        
        history.append({
            'epoch': epoch,
            'train_loss': float(avg_loss),
            'val_pcc': float(val_pcc),
            'lr': float(scheduler.get_last_lr()[0])
        })
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  loss={avg_loss:.4f}  val_PCC={val_pcc:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")
        
        if val_pcc > best_val_pcc:
            best_val_pcc = val_pcc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break
    
    # Final test
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    
    test_metrics = evaluate_full(model, snp_data, phenotype, test_idx, device)
    val_metrics = evaluate_full(model, snp_data, phenotype, val_idx, device)
    
    print(f"\n  Final Results:")
    print(f"    Val  PCC: {val_metrics['pearson_avg']:.4f}  Spearman: {val_metrics['spearman_avg']:.4f}")
    print(f"    Test PCC: {test_metrics['pearson_avg']:.4f}  Spearman: {test_metrics['spearman_avg']:.4f}")
    
    return model, best_state, {
        'model_name': model_name,
        'n_params': n_params,
        'best_val_pcc': float(best_val_pcc),
        'test_metrics': test_metrics,
        'val_metrics': val_metrics,
        'history': history,
        'epochs_trained': epoch,
    }


def main():
    print("=" * 80)
    print("Full Training — Best Models from Diagnosis")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    data_dir = Path("data/processed/GSTP007_full_10000snps_processed")
    
    snp_data = torch.from_numpy(np.load(data_dir / "genotype_onehot.npy")).float()
    phenotype = torch.from_numpy(np.load(data_dir / "phenotype_scaled.npy")).float()
    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)
    with open(data_dir / "split.json") as f:
        splits = json.load(f)
    
    n_snps = meta['n_snps_selected']
    n_traits = meta['n_traits']
    
    print(f"SNPs: {n_snps:,}  Samples: {meta['n_samples']}  Traits: {n_traits}")
    print(f"Train/Val/Test: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    
    all_results = {}
    
    # --- Model 1: V2 Anchor Similarity (best from diagnosis) ---
    model_v2 = AnchorSimilarityModel(n_snps, n_traits, n_anchors=200, d_hidden=256, dropout=0.3)
    model_v2, state_v2, res_v2 = train_model(
        model_v2, snp_data, phenotype, splits, device,
        n_epochs=60, lr=0.001, weight_decay=1e-4, batch_size=32,
        model_name="V2-AnchorSimilarity"
    )
    all_results['V2_anchor'] = res_v2
    torch.save(state_v2, data_dir / "best_v2_anchor_model.pt")
    
    # --- Model 2: V1 Multi-scale CNN ---
    model_v1 = MultiScaleCNNModel(n_snps, n_traits, d_hidden=256, dropout=0.3)
    model_v1, state_v1, res_v1 = train_model(
        model_v1, snp_data, phenotype, splits, device,
        n_epochs=60, lr=0.001, weight_decay=1e-4, batch_size=32,
        model_name="V1-MultiScaleCNN"
    )
    all_results['V1_cnn'] = res_v1
    torch.save(state_v1, data_dir / "best_v1_cnn_model.pt")
    
    # --- Model 3: Ensemble of V1 + V2 ---
    print(f"\n{'='*70}")
    print("Ensemble: V1 + V2 (average predictions)")
    print(f"{'='*70}")
    
    test_idx = splits['test']
    model_v2.to(device).eval()
    model_v1.to(device).eval()
    
    with torch.no_grad():
        preds_v2_list, preds_v1_list = [], []
        for i in range(0, len(test_idx), 64):
            batch_idx = test_idx[i:i+64]
            x = snp_data[batch_idx].to(device)
            preds_v2_list.append(model_v2(x).cpu())
            preds_v1_list.append(model_v1(x).cpu())
        preds_v2 = torch.cat(preds_v2_list).numpy()
        preds_v1 = torch.cat(preds_v1_list).numpy()
    
    targets = phenotype[test_idx].numpy()
    
    # Search best ensemble weight on val set
    val_idx = splits['val']
    with torch.no_grad():
        vpreds_v2_list, vpreds_v1_list = [], []
        for i in range(0, len(val_idx), 64):
            batch_idx = val_idx[i:i+64]
            x = snp_data[batch_idx].to(device)
            vpreds_v2_list.append(model_v2(x).cpu())
            vpreds_v1_list.append(model_v1(x).cpu())
        vpreds_v2 = torch.cat(vpreds_v2_list).numpy()
        vpreds_v1 = torch.cat(vpreds_v1_list).numpy()
    
    val_targets = phenotype[val_idx].numpy()
    
    best_w, best_val_ens_pcc = 0.5, -999
    for w in np.arange(0.0, 1.05, 0.05):
        ens = w * vpreds_v2 + (1 - w) * vpreds_v1
        pccs = []
        for t in range(n_traits):
            mask = ~np.isnan(val_targets[:, t])
            if mask.sum() > 10 and np.std(ens[mask, t]) > 1e-8:
                pcc, _ = pearsonr(val_targets[mask, t], ens[mask, t])
                pccs.append(pcc)
        avg = np.mean(pccs) if pccs else -999
        if avg > best_val_ens_pcc:
            best_val_ens_pcc = avg
            best_w = w
    
    print(f"  Best ensemble weight: V2={best_w:.2f}, V1={1-best_w:.2f}")
    print(f"  Val ensemble PCC: {best_val_ens_pcc:.4f}")
    
    # Test ensemble
    ens_test = best_w * preds_v2 + (1 - best_w) * preds_v1
    ens_pccs = []
    for t in range(n_traits):
        mask = ~np.isnan(targets[:, t])
        if mask.sum() > 10 and np.std(ens_test[mask, t]) > 1e-8:
            pcc, _ = pearsonr(targets[mask, t], ens_test[mask, t])
            ens_pccs.append(float(pcc))
        else:
            ens_pccs.append(0.0)
    
    ens_pcc = float(np.mean(ens_pccs))
    print(f"  Test ensemble PCC: {ens_pcc:.4f}")
    
    all_results['ensemble_v1v2'] = {
        'model_name': f'Ensemble(V2*{best_w:.2f}+V1*{1-best_w:.2f})',
        'test_pcc': ens_pcc,
        'val_pcc': float(best_val_ens_pcc),
        'weight_v2': float(best_w),
    }
    
    # === Final Summary ===
    print(f"\n{'='*80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Model':<40} {'Test PCC':>10} {'Val PCC':>10} {'Params':>12}")
    print("-" * 75)
    print(f"{'GBLUP (reference)':<40} {'0.538':>10} {'—':>10} {'—':>12}")
    print(f"{'Original PlantHGNN (before fix)':<40} {'0.233':>10} {'0.238':>10} {'1,235,459':>12}")
    
    for name, r in all_results.items():
        label = r.get('model_name', name)
        test_pcc = r.get('test_metrics', {}).get('pearson_avg', r.get('test_pcc', 0))
        val_pcc = r.get('val_metrics', {}).get('pearson_avg', r.get('val_pcc', r.get('best_val_pcc', 0)))
        n_params = r.get('n_params', '—')
        params_str = f"{n_params:,}" if isinstance(n_params, int) else str(n_params)
        print(f"{label:<40} {test_pcc:>10.4f} {val_pcc:>10.4f} {params_str:>12}")
    
    # Save all results
    serializable = {}
    for k, v in all_results.items():
        serializable[k] = {}
        for kk, vv in v.items():
            if isinstance(vv, (int, float, str, bool)):
                serializable[k][kk] = vv
            elif isinstance(vv, dict):
                serializable[k][kk] = {kkk: float(vvv) if isinstance(vvv, (np.floating, float)) else
                                        [float(x) for x in vvv] if isinstance(vvv, list) else vvv
                                        for kkk, vvv in vv.items()}
            elif isinstance(vv, list):
                serializable[k][kk] = [float(x) if isinstance(x, (np.floating, float)) else x for x in vv]
    
    with open(data_dir / "best_model_results.json", 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    
    print(f"\n✓ Models saved to {data_dir}")
    print(f"✓ Results saved to {data_dir / 'best_model_results.json'}")
    
    return 0


if __name__ == '__main__':
    exit(main())

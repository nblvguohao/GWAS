#!/usr/bin/env python3
"""
在 GSTP007 真实水稻数据上训练 PlantHGNN 并与基线对比。

基线：
  - GBLUP (Ridge Regression on top-5000 PCS SNPs)
  - MLP   (3层MLP)

PlantHGNN变体：
  - Transformer-only (无GCN, 无AttnRes)
  - Transformer + AttnRes
  - (未来) + MultiViewGCN

评估指标：Pearson correlation coefficient (PCC), MSE
"""

import sys
sys.path.insert(0, 'E:/GWAS')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from pathlib import Path
import json
import logging
import time

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

from src.models.plant_hgnn import PlantHGNN

# ── 配置 ────────────────────────────────────────────────────────────────────────
PROC_DIR = Path('E:/GWAS/data/processed/gstp007')
RESULT_DIR = Path('E:/GWAS/results/gstp007')
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"设备: {DEVICE}")

TRAITS = [
    'Plant_Height', 'Grain_Length', 'Grain_Width',
    'Days_to_Heading', 'Panicle_Length', 'Grain_Weight', 'Yield_per_plant',
]

# 训练超参数
TRAIN_CFG = dict(
    batch_size  = 64,
    lr          = 5e-4,
    weight_decay= 1e-4,
    max_epochs  = 200,
    patience    = 25,
    d_model     = 128,
    n_layers    = 6,
    n_blocks    = 8,
    n_heads     = 8,
    dropout     = 0.2,
)


# ── Dataset ──────────────────────────────────────────────────────────────────────
class SNPDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ── Baselines ─────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, n_snps: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_snps, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256),   nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128),   nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


def run_gblup(X_train, y_train, X_test, alpha=10.0):
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_train, y_train)
    return model.predict(X_test)


# ── 训练循环 ──────────────────────────────────────────────────────────────────────
def train_nn_model(model, train_loader, val_loader,
                   lr=1e-3, weight_decay=1e-4,
                   max_epochs=200, patience=25):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=lr * 0.01
    )

    best_val_loss = float('inf')
    best_state    = None
    no_improve    = 0

    for epoch in range(max_epochs):
        # Train
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = F.mse_loss(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validate
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                pred = model(X_batch).cpu().numpy()
                val_preds.extend(pred)
                val_targets.extend(y_batch.numpy())

        val_loss = np.mean((np.array(val_preds) - np.array(val_targets)) ** 2)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            logger.info(f"  早停 @ epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate_model_nn(model, X_test, y_test) -> dict:
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        preds = model(X_t).cpu().numpy()
    pcc, _ = pearsonr(y_test, preds)
    mse    = np.mean((y_test - preds) ** 2)
    return {'pcc': float(pcc), 'mse': float(mse)}


# ── 主训练循环 ────────────────────────────────────────────────────────────────────
def run_trait(trait: str, cfg: dict) -> dict:
    trait_dir = PROC_DIR / trait
    if not trait_dir.exists():
        logger.warning(f"跳过 {trait}: 目录不存在")
        return {}

    X_train = np.load(trait_dir / 'X_train.npy')
    X_val   = np.load(trait_dir / 'X_val.npy')
    X_test  = np.load(trait_dir / 'X_test.npy')
    y_train = np.load(trait_dir / 'y_train_scaled.npy')
    y_val   = np.load(trait_dir / 'y_val_scaled.npy')
    y_test  = np.load(trait_dir / 'y_test_scaled.npy')

    n_snps = X_train.shape[1]
    results = {}

    print(f"\n{'─'*55}")
    print(f"  性状: {trait}  |  训练样本: {len(X_train)}")
    print(f"{'─'*55}")

    # ── GBLUP ────────────────────────────────────────────────────────────────
    t0 = time.time()
    gblup_preds = run_gblup(X_train, y_train, X_test)
    pcc_g, _ = pearsonr(y_test, gblup_preds)
    mse_g    = np.mean((y_test - gblup_preds) ** 2)
    results['GBLUP'] = {'pcc': float(pcc_g), 'mse': float(mse_g)}
    print(f"  GBLUP:          PCC={pcc_g:+.4f}  MSE={mse_g:.4f}  [{time.time()-t0:.1f}s]")

    bs = cfg['batch_size']
    train_ds = SNPDataset(X_train, y_train)
    val_ds   = SNPDataset(X_val,   y_val)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=0)

    # ── MLP ──────────────────────────────────────────────────────────────────
    t0 = time.time()
    mlp = MLP(n_snps, cfg['dropout']).to(DEVICE)
    mlp = train_nn_model(mlp, train_loader, val_loader,
                          lr=cfg['lr'], weight_decay=cfg['weight_decay'],
                          max_epochs=cfg['max_epochs'], patience=cfg['patience'])
    mlp_res = evaluate_model_nn(mlp, X_test, y_test)
    results['MLP'] = mlp_res
    print(f"  MLP:            PCC={mlp_res['pcc']:+.4f}  MSE={mlp_res['mse']:.4f}  "
          f"[{time.time()-t0:.1f}s]")

    # ── PlantHGNN (Transformer Only, 无GCN) ──────────────────────────────────
    t0 = time.time()
    model_no_attnres = PlantHGNN(
        n_snps    = n_snps,
        d_model   = cfg['d_model'],
        n_transformer_layers = cfg['n_layers'],
        n_attnres_blocks     = cfg['n_blocks'],
        n_traits  = 1,
        use_gcn   = False,
        use_attnres = False,
        n_heads   = cfg['n_heads'],
        dropout   = cfg['dropout'],
    ).to(DEVICE)
    model_no_attnres = train_nn_model(
        model_no_attnres, train_loader, val_loader,
        lr=cfg['lr'], weight_decay=cfg['weight_decay'],
        max_epochs=cfg['max_epochs'], patience=cfg['patience']
    )
    res_no_ar = evaluate_model_nn(model_no_attnres, X_test, y_test)
    results['Transformer'] = res_no_ar
    print(f"  Transformer:    PCC={res_no_ar['pcc']:+.4f}  MSE={res_no_ar['mse']:.4f}  "
          f"[{time.time()-t0:.1f}s]")

    # ── PlantHGNN (Transformer + AttnRes) ────────────────────────────────────
    t0 = time.time()
    model_attnres = PlantHGNN(
        n_snps    = n_snps,
        d_model   = cfg['d_model'],
        n_transformer_layers = cfg['n_layers'],
        n_attnres_blocks     = cfg['n_blocks'],
        n_traits  = 1,
        use_gcn   = False,
        use_attnres = True,
        n_heads   = cfg['n_heads'],
        dropout   = cfg['dropout'],
    ).to(DEVICE)
    model_attnres = train_nn_model(
        model_attnres, train_loader, val_loader,
        lr=cfg['lr'], weight_decay=cfg['weight_decay'],
        max_epochs=cfg['max_epochs'], patience=cfg['patience']
    )
    res_ar = evaluate_model_nn(model_attnres, X_test, y_test)
    results['PlantHGNN_AttnRes'] = res_ar
    delta_attnres = res_ar['pcc'] - res_no_ar['pcc']
    print(f"  PlantHGNN+AttnRes: PCC={res_ar['pcc']:+.4f}  MSE={res_ar['mse']:.4f}  "
          f"[AttnRes Δ={delta_attnres:+.4f}]  [{time.time()-t0:.1f}s]")

    return results


def main():
    print("=" * 60)
    print("GSTP007 真实数据训练：PlantHGNN vs Baselines")
    print("=" * 60)

    all_results = {}
    cfg = TRAIN_CFG

    for trait in TRAITS:
        results = run_trait(trait, cfg)
        if results:
            all_results[trait] = results

    # ── 汇总表 ─────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("汇总：各性状 Pearson PCC")
    print("=" * 70)
    models = ['GBLUP', 'MLP', 'Transformer', 'PlantHGNN_AttnRes']
    header = f"{'性状':<22}" + "".join(f"{m:>18}" for m in models)
    print(header)
    print("-" * 70)

    avg_pccs = {m: [] for m in models}
    for trait, res in all_results.items():
        row = f"{trait:<22}"
        for m in models:
            if m in res:
                pcc = res[m]['pcc']
                avg_pccs[m].append(pcc)
                row += f"{pcc:>+18.4f}"
            else:
                row += f"{'N/A':>18}"
        print(row)

    print("-" * 70)
    avg_row = f"{'AVERAGE':<22}"
    for m in models:
        if avg_pccs[m]:
            avg_row += f"{np.mean(avg_pccs[m]):>+18.4f}"
        else:
            avg_row += f"{'N/A':>18}"
    print(avg_row)

    # 保存结果
    out_path = RESULT_DIR / 'gstp007_comparison.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\n结果保存至: {out_path}")

    # 关键判断
    avg_attnres = np.mean(avg_pccs.get('PlantHGNN_AttnRes', [0]))
    avg_gblup   = np.mean(avg_pccs.get('GBLUP', [0]))
    avg_trans   = np.mean(avg_pccs.get('Transformer', [0]))
    print(f"\n=== 模型评估 ===")
    print(f"PlantHGNN+AttnRes vs GBLUP:        Δ={avg_attnres - avg_gblup:+.4f}")
    print(f"PlantHGNN+AttnRes vs Transformer:  Δ={avg_attnres - avg_trans:+.4f}")
    if avg_attnres > avg_gblup + 0.02:
        print("✓ 深度学习模型显著优于GBLUP → 继续推进GCN集成")
    else:
        print("⚠ 深度学习模型未能显著超越GBLUP → 需要加入图结构信息（MultiViewGCN）")


if __name__ == '__main__':
    main()

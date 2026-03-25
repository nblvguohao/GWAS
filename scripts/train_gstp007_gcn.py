#!/usr/bin/env python3
"""
GSTP007完整对比实验: GBLUP vs MLP vs Transformer vs PlantHGNN(+GCN+AttnRes)

PlantHGNN变体：
  1. Transformer Only
  2. Transformer + AttnRes
  3. Transformer + GCN (无AttnRes)
  4. PlantHGNN Full (Transformer + GCN + AttnRes)
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
import scipy.sparse as sp
from pathlib import Path
import json
import time
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

from src.models.plant_hgnn import PlantHGNN

# ── 路径 ───────────────────────────────────────────────────────────────────────
PROC_DIR  = Path('E:/GWAS/data/processed/gstp007')
GRAPH_DIR = PROC_DIR / 'graph'
RESULT_DIR = Path('E:/GWAS/results/gstp007')
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAITS = [
    'Plant_Height', 'Grain_Length', 'Grain_Width',
    'Days_to_Heading', 'Panicle_Length', 'Grain_Weight', 'Yield_per_plant',
]

CFG = dict(
    batch_size   = 64,
    lr           = 5e-4,
    weight_decay = 1e-4,
    max_epochs   = 200,
    patience     = 25,
    d_model      = 128,
    n_layers     = 6,
    n_blocks     = 8,
    n_heads      = 8,
    dropout      = 0.2,
)


# ── Dataset（支持基因特征）────────────────────────────────────────────────────
class GPDataset(Dataset):
    def __init__(self, X_snp: np.ndarray, y: np.ndarray,
                 X_gene: np.ndarray = None):
        self.X_snp  = torch.tensor(X_snp,  dtype=torch.float32)
        self.y      = torch.tensor(y,       dtype=torch.float32)
        self.X_gene = torch.tensor(X_gene, dtype=torch.float32) \
                      if X_gene is not None else None

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        if self.X_gene is not None:
            return self.X_snp[i], self.y[i], self.X_gene[i]
        return self.X_snp[i], self.y[i]


def collate_with_gene(batch):
    if len(batch[0]) == 3:
        snp, y, gene = zip(*batch)
        return torch.stack(snp), torch.stack(y), torch.stack(gene)
    snp, y = zip(*batch)
    return torch.stack(snp), torch.stack(y)


# ── 训练函数 ──────────────────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, adj_gpu=None, cfg=CFG):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['max_epochs'], eta_min=cfg['lr'] * 0.01
    )

    best_val_loss = float('inf')
    best_state    = None
    no_improve    = 0

    for epoch in range(cfg['max_epochs']):
        model.train()
        for batch in train_loader:
            if len(batch) == 3:
                X_snp, y_batch, X_gene = batch
                X_snp   = X_snp.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                X_gene  = X_gene.to(DEVICE)
            else:
                X_snp, y_batch = batch
                X_snp   = X_snp.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                X_gene  = None

            optimizer.zero_grad()
            adj_list = [adj_gpu, None, None] if adj_gpu is not None else None
            pred = model(X_snp, X_gene, adj_list)
            loss = F.mse_loss(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    X_snp, y_batch, X_gene = batch
                    X_snp, X_gene = X_snp.to(DEVICE), X_gene.to(DEVICE)
                else:
                    X_snp, y_batch = batch
                    X_snp  = X_snp.to(DEVICE)
                    X_gene = None
                adj_list = [adj_gpu, None, None] if adj_gpu is not None else None
                pred = model(X_snp, X_gene, adj_list).cpu()
                val_losses.append(F.mse_loss(pred, y_batch).item())

        val_loss = np.mean(val_losses)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= cfg['patience']:
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate(model, X_snp, y, adj_gpu=None, X_gene=None) -> dict:
    model.eval()
    X_t = torch.tensor(X_snp, dtype=torch.float32).to(DEVICE)
    g_t = torch.tensor(X_gene, dtype=torch.float32).to(DEVICE) \
          if X_gene is not None else None
    adj_list = [adj_gpu, None, None] if adj_gpu is not None else None
    with torch.no_grad():
        preds = model(X_t, g_t, adj_list).cpu().numpy()
    pcc, _ = pearsonr(y, preds)
    mse    = np.mean((y - preds) ** 2)
    return {'pcc': float(pcc), 'mse': float(mse)}


# ── 加载共享图数据 ─────────────────────────────────────────────────────────────
def load_graph():
    adj_path  = GRAPH_DIR / 'ppi_adj.npz'
    gene_path = GRAPH_DIR / 'gene_list.txt'
    if not adj_path.exists():
        logger.warning("PPI邻接矩阵未找到，跳过GCN模型")
        return None, None, 0

    adj_sp   = sp.load_npz(adj_path)
    adj_dense = torch.tensor(adj_sp.toarray(), dtype=torch.float32).to(DEVICE)
    gene_list = gene_path.read_text().strip().split('\n')
    n_genes   = len(gene_list)
    logger.info(f"加载PPI图: {adj_dense.shape}, {n_genes} 基因")
    return adj_dense, gene_list, n_genes


# ── 主循环 ─────────────────────────────────────────────────────────────────────
def run_trait(trait: str, adj_gpu, n_genes: int, cfg=CFG) -> dict:
    trait_dir      = PROC_DIR / trait
    trait_graph_dir = GRAPH_DIR / trait

    X_train = np.load(trait_dir / 'X_train.npy')
    X_val   = np.load(trait_dir / 'X_val.npy')
    X_test  = np.load(trait_dir / 'X_test.npy')
    y_train = np.load(trait_dir / 'y_train_scaled.npy')
    y_val   = np.load(trait_dir / 'y_val_scaled.npy')
    y_test  = np.load(trait_dir / 'y_test_scaled.npy')
    n_snps  = X_train.shape[1]

    has_gene = (trait_graph_dir / 'gene_feat_train.npy').exists() and n_genes > 0
    if has_gene:
        G_train = np.load(trait_graph_dir / 'gene_feat_train.npy')
        G_val   = np.load(trait_graph_dir / 'gene_feat_val.npy')
        G_test  = np.load(trait_graph_dir / 'gene_feat_test.npy')
    else:
        G_train = G_val = G_test = None

    bs = cfg['batch_size']
    train_ds = GPDataset(X_train, y_train, G_train)
    val_ds   = GPDataset(X_val,   y_val,   G_val)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                               collate_fn=collate_with_gene, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                               collate_fn=collate_with_gene, num_workers=0)

    results = {}
    print(f"\n{'─'*62}")
    print(f"  {trait}  [train={len(X_train)}, SNPs={n_snps}, genes={n_genes if has_gene else 'N/A'}]")
    print(f"{'─'*62}")

    # 1. GBLUP
    t0 = time.time()
    gblup_preds = Ridge(alpha=10.0).fit(X_train, y_train).predict(X_test)
    pcc_g, _ = pearsonr(y_test, gblup_preds)
    results['GBLUP'] = {'pcc': float(pcc_g), 'mse': float(np.mean((y_test-gblup_preds)**2))}
    print(f"  GBLUP:              PCC={pcc_g:+.4f}  [{time.time()-t0:.1f}s]")

    # 2. Transformer Only
    t0 = time.time()
    m = PlantHGNN(n_snps, d_model=cfg['d_model'], n_transformer_layers=cfg['n_layers'],
                  n_attnres_blocks=cfg['n_blocks'], use_gcn=False, use_attnres=False,
                  n_heads=cfg['n_heads'], dropout=cfg['dropout']).to(DEVICE)
    m = train_model(m, train_loader, val_loader, adj_gpu=None, cfg=cfg)
    res = evaluate(m, X_test, y_test)
    results['Transformer'] = res
    print(f"  Transformer:        PCC={res['pcc']:+.4f}  [{time.time()-t0:.1f}s]")

    # 3. Transformer + AttnRes
    t0 = time.time()
    m = PlantHGNN(n_snps, d_model=cfg['d_model'], n_transformer_layers=cfg['n_layers'],
                  n_attnres_blocks=cfg['n_blocks'], use_gcn=False, use_attnres=True,
                  n_heads=cfg['n_heads'], dropout=cfg['dropout']).to(DEVICE)
    m = train_model(m, train_loader, val_loader, adj_gpu=None, cfg=cfg)
    res = evaluate(m, X_test, y_test)
    results['Transformer+AttnRes'] = res
    print(f"  Transformer+AttnRes: PCC={res['pcc']:+.4f}  [{time.time()-t0:.1f}s]")

    if has_gene and adj_gpu is not None:
        # 4. GCN Only (no AttnRes)
        t0 = time.time()
        m = PlantHGNN(n_snps, d_model=cfg['d_model'], n_transformer_layers=cfg['n_layers'],
                      n_attnres_blocks=cfg['n_blocks'],
                      n_gcn_genes=n_genes, n_views=3,
                      use_gcn=True, use_attnres=False,
                      n_heads=cfg['n_heads'], dropout=cfg['dropout']).to(DEVICE)
        m = train_model(m, train_loader, val_loader, adj_gpu=adj_gpu, cfg=cfg)
        res = evaluate(m, X_test, y_test, adj_gpu=adj_gpu, X_gene=G_test)
        results['Transformer+GCN'] = res
        print(f"  Transformer+GCN:    PCC={res['pcc']:+.4f}  [{time.time()-t0:.1f}s]")

        # 5. PlantHGNN Full
        t0 = time.time()
        m = PlantHGNN(n_snps, d_model=cfg['d_model'], n_transformer_layers=cfg['n_layers'],
                      n_attnres_blocks=cfg['n_blocks'],
                      n_gcn_genes=n_genes, n_views=3,
                      use_gcn=True, use_attnres=True,
                      n_heads=cfg['n_heads'], dropout=cfg['dropout']).to(DEVICE)
        m = train_model(m, train_loader, val_loader, adj_gpu=adj_gpu, cfg=cfg)
        res = evaluate(m, X_test, y_test, adj_gpu=adj_gpu, X_gene=G_test)
        results['PlantHGNN_Full'] = res
        delta = res['pcc'] - results['Transformer+AttnRes']['pcc']
        print(f"  PlantHGNN_Full:     PCC={res['pcc']:+.4f}  "
              f"[GCN delta={delta:+.4f}]  [{time.time()-t0:.1f}s]")

    return results


def main():
    print("=" * 65)
    print("GSTP007 完整对比: GBLUP vs MLP vs PlantHGNN变体")
    print("=" * 65)

    adj_gpu, gene_list, n_genes = load_graph()
    all_results = {}

    for trait in TRAITS:
        if not (PROC_DIR / trait).exists():
            continue
        all_results[trait] = run_trait(trait, adj_gpu, n_genes, CFG)

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    models = ['GBLUP', 'Transformer', 'Transformer+AttnRes',
              'Transformer+GCN', 'PlantHGNN_Full']

    print("\n\n" + "=" * 75)
    print("汇总 PCC (Pearson)")
    print("=" * 75)
    header = f"{'性状':<22}" + "".join(f"{m[:14]:>14}" for m in models)
    print(header)
    print("-" * 75)

    avgs = {m: [] for m in models}
    for trait, res in all_results.items():
        row = f"{trait:<22}"
        for m in models:
            if m in res:
                pcc = res[m]['pcc']
                avgs[m].append(pcc)
                row += f"{pcc:>+14.4f}"
            else:
                row += f"{'—':>14}"
        print(row)

    print("-" * 75)
    avg_row = f"{'AVERAGE':<22}"
    for m in models:
        if avgs[m]:
            avg_row += f"{np.mean(avgs[m]):>+14.4f}"
        else:
            avg_row += f"{'—':>14}"
    print(avg_row)

    # 保存
    out = RESULT_DIR / 'gstp007_full_comparison.json'
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果保存: {out}")

    # 判断
    avg_full  = np.mean(avgs.get('PlantHGNN_Full', [0]))
    avg_gblup = np.mean(avgs.get('GBLUP', [0]))
    avg_trans = np.mean(avgs.get('Transformer', [0]))
    print(f"\n=== 关键结论 ===")
    print(f"PlantHGNN_Full vs GBLUP:       delta={avg_full - avg_gblup:+.4f}")
    print(f"PlantHGNN_Full vs Transformer: delta={avg_full - avg_trans:+.4f}")


if __name__ == '__main__':
    main()

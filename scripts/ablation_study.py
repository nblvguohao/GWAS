#!/usr/bin/env python3
"""
PlantHGNN 消融实验（Ablation Study）

消融组：
  A: 网络/GCN相关
    - no_GCN:        PlantHGNN without GCN (only Transformer+AttnRes)
    - single_view:   PlantHGNN with single GCN view (no multi-view fusion)

  B: AttnRes相关
    - no_AttnRes:    PlantHGNN without AttnRes (standard residual Transformer)
    - std_residual:  Same as no_AttnRes (for clarity)

  C: 融合策略相关
    - concat_fusion: 直接拼接三视图特征（固定权重）
    - mean_fusion:   三视图取平均
    - attn_fusion:   可学习注意力融合（完整版 = PlantHGNN_Full）

  D: 完整模型
    - PlantHGNN_Full: 完整模型（对照组）

输出：
  results/gstp007/ablation_results.json
  results/gstp007/ablation_table.csv
"""

import sys
sys.path.insert(0, 'E:/GWAS')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
from pathlib import Path
import json
import time
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('E:/GWAS/results/gstp007/ablation_study.log',
                            mode='w', encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)

from src.models.plant_hgnn import PlantHGNN
from src.training.metrics import wilcoxon_test

PROC_DIR   = Path('E:/GWAS/data/processed/gstp007')
GRAPH_DIR  = PROC_DIR / 'graph'
RESULT_DIR = Path('E:/GWAS/results/gstp007')
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAITS  = ['Plant_Height', 'Grain_Length', 'Grain_Width',
           'Days_to_Heading', 'Panicle_Length', 'Grain_Weight', 'Yield_per_plant']
N_FOLDS = 5
SEEDS   = [42, 123]
TOP_K   = 5000

CFG = dict(
    batch_size=64, lr=5e-4, weight_decay=1e-4,
    max_epochs=150, patience=20,
    d_model=128, n_layers=6, n_blocks=8, n_heads=8, dropout=0.2,
)

# ── 消融配置 ──────────────────────────────────────────────────────────────────
ABLATION_CONFIGS = {
    # 完整模型
    'PlantHGNN_Full':      dict(use_gcn=True,  use_attnres=True),
    # AttnRes 消融
    'no_AttnRes':          dict(use_gcn=True,  use_attnres=False),
    # GCN 消融
    'no_GCN':              dict(use_gcn=False, use_attnres=True),
    # 两者都去掉（纯Transformer基线）
    'Transformer_Only':    dict(use_gcn=False, use_attnres=False),
}


class GPDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, G=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.G = torch.tensor(G, dtype=torch.float32) if G is not None else None

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        if self.G is not None:
            return self.X[i], self.y[i], self.G[i]
        return self.X[i], self.y[i]


def pcs_select(X_tr, y_tr, top_k=TOP_K):
    y_c   = y_tr - y_tr.mean()
    y_n   = np.sqrt((y_c**2).sum()) + 1e-8
    X_c   = X_tr - X_tr.mean(0)
    r     = (X_c * y_c[:, None]).sum(0) / (np.sqrt((X_c**2).sum(0)) + 1e-8) / y_n
    k     = min(top_k, X_tr.shape[1])
    idx   = np.argpartition(np.abs(r), -k)[-k:]
    return idx


def train_and_eval(X_tr, y_tr, X_va, y_va, X_te, y_te,
                   g_tr, g_va, g_te, adj, cfg, use_gcn, use_attnres):
    n_snps = X_tr.shape[1]
    n_genes = g_tr.shape[1] if (g_tr is not None and use_gcn) else 0

    model = PlantHGNN(
        n_snps=n_snps, d_model=cfg['d_model'],
        n_transformer_layers=cfg['n_layers'],
        n_attnres_blocks=cfg['n_blocks'], n_traits=1,
        n_gcn_genes=n_genes, n_views=1,
        use_gcn=use_gcn and (n_genes > 0),
        use_attnres=use_attnres,
        n_heads=cfg['n_heads'], dropout=cfg['dropout'],
    ).to(DEVICE)

    opt = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['max_epochs'], eta_min=1e-5)

    use_gene = use_gcn and g_tr is not None and n_genes > 0
    adj_dev  = None
    if use_gene and adj is not None:
        adj_dev = [torch.tensor(adj, dtype=torch.float32).to(DEVICE)]

    dl_tr = DataLoader(GPDataset(X_tr, y_tr, g_tr if use_gene else None),
                       batch_size=cfg['batch_size'], shuffle=True)

    best_pcc, best_state, no_imp = -1.0, None, 0
    for epoch in range(cfg['max_epochs']):
        model.train()
        for batch in dl_tr:
            if len(batch) == 3:
                xb, yb, gb = [t.to(DEVICE) for t in batch]
            else:
                xb, yb = [t.to(DEVICE) for t in batch]; gb = None
            loss = nn.MSELoss()(model(xb, gene_feat=gb, adj_list=adj_dev), yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        Xv = torch.tensor(X_va, dtype=torch.float32).to(DEVICE)
        Gv = torch.tensor(g_va, dtype=torch.float32).to(DEVICE) if use_gene else None
        with torch.no_grad():
            pv = model(Xv, gene_feat=Gv, adj_list=adj_dev).cpu().numpy()
        pcc = pearsonr(y_va, pv)[0]
        if pcc > best_pcc:
            best_pcc = pcc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= cfg['patience']:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    Xt = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
    Gt = torch.tensor(g_te, dtype=torch.float32).to(DEVICE) if use_gene else None
    with torch.no_grad():
        pt = model(Xt, gene_feat=Gt, adj_list=adj_dev).cpu().numpy()
    pcc = pearsonr(y_te, pt)[0]
    scc = spearmanr(y_te, pt)[0]
    del model; torch.cuda.empty_cache()
    return float(pcc), float(scc)


def load_trait_data(trait):
    td = PROC_DIR / trait
    X_tr = np.load(td / 'X_train.npy'); X_va = np.load(td / 'X_val.npy')
    X_te = np.load(td / 'X_test.npy')
    y_tr = np.load(td / 'y_train.npy'); y_va = np.load(td / 'y_val.npy')
    y_te = np.load(td / 'y_test.npy')
    X_all = np.concatenate([X_tr, X_va, X_te]); y_all = np.concatenate([y_tr, y_va, y_te])

    gene_all = None
    gd = GRAPH_DIR / trait
    if gd.exists():
        parts = []
        for split in ['train', 'val', 'test']:
            p = gd / f'gene_feat_{split}.npy'
            if p.exists(): parts.append(np.load(str(p)))
        if len(parts) == 3: gene_all = np.concatenate(parts)

    adj = None
    for ap in [GRAPH_DIR / 'adj_norm.npy', GRAPH_DIR / 'adj_norm.npz',
               GRAPH_DIR / 'ppi_adj.npz', GRAPH_DIR / 'ppi_adj.npy']:
        if ap.exists():
            adj = np.load(str(ap)) if ap.suffix == '.npy' \
                  else sp.load_npz(str(ap)).toarray().astype(np.float32)
            break
    return X_all, y_all, gene_all, adj


def main():
    logger.info("=" * 60)
    logger.info("PlantHGNN Ablation Study")
    logger.info("=" * 60)

    all_results = {}

    for trait in TRAITS:
        logger.info(f"\n[{trait}]")
        try:
            X_all, y_all, gene_all, adj = load_trait_data(trait)
        except Exception as e:
            logger.warning(f"  skip: {e}"); continue

        all_results[trait] = {k: [] for k in ABLATION_CONFIGS}

        for seed in SEEDS:
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            for fold, (tr_idx, te_idx) in enumerate(kf.split(X_all)):
                torch.manual_seed(seed + fold)
                np.random.seed(seed + fold)

                n_val = max(int(len(tr_idx) * 0.15), 1)
                rng = np.random.RandomState(seed + fold)
                va_loc = rng.choice(len(tr_idx), n_val, replace=False)
                tr_loc = np.setdiff1d(np.arange(len(tr_idx)), va_loc)
                atr = tr_idx[tr_loc]; ava = tr_idx[va_loc]; ate = te_idx

                # PCS
                snp_idx = pcs_select(X_all[atr], y_all[atr])
                scl = StandardScaler()
                X_tr = scl.fit_transform(X_all[atr][:, snp_idx])
                X_va = scl.transform(X_all[ava][:, snp_idx])
                X_te = scl.transform(X_all[ate][:, snp_idx])

                ym, ys = y_all[atr].mean(), y_all[atr].std() + 1e-8
                y_tr = (y_all[atr] - ym) / ys
                y_va = (y_all[ava] - ym) / ys
                y_te_raw = y_all[ate]

                if gene_all is not None:
                    gs = StandardScaler()
                    g_tr = gs.fit_transform(gene_all[atr])
                    g_va = gs.transform(gene_all[ava])
                    g_te = gs.transform(gene_all[ate])
                else:
                    g_tr = g_va = g_te = None

                for config_name, abl_cfg in ABLATION_CONFIGS.items():
                    skip_gcn = not abl_cfg['use_gcn']
                    if not skip_gcn and (g_tr is None or adj is None):
                        skip_gcn = True
                    try:
                        pcc, scc = train_and_eval(
                            X_tr, y_tr, X_va, y_va, X_te,
                            (y_te_raw - ym) / ys,  # test in scaled space for model
                            g_tr if not skip_gcn else None,
                            g_va if not skip_gcn else None,
                            g_te if not skip_gcn else None,
                            adj if not skip_gcn else None,
                            CFG,
                            use_gcn=not skip_gcn,
                            use_attnres=abl_cfg['use_attnres'],
                        )
                        # Convert PCC back: model predicts scaled y, correlation is invariant to linear transform
                        all_results[trait][config_name].append({
                            'fold': fold, 'seed': seed, 'pcc': pcc, 'spearman': scc,
                        })
                        logger.info(f"  {config_name:<22} fold{fold}/seed{seed}: PCC={pcc:.4f}")
                    except Exception as e:
                        logger.warning(f"  {config_name} fold{fold}/seed{seed} failed: {e}")

    # ── 保存 + 汇总 ──────────────────────────────────────────────────────────
    with open(RESULT_DIR / 'ablation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Build table
    rows = []
    for trait in TRAITS:
        if trait not in all_results: continue
        row = {'Trait': trait}
        for cfg_name in ABLATION_CONFIGS:
            runs = all_results[trait].get(cfg_name, [])
            if runs:
                pccs = [r['pcc'] for r in runs]
                row[cfg_name] = f"{np.mean(pccs):.4f}±{np.std(pccs):.4f}"
            else:
                row[cfg_name] = '-'
        rows.append(row)

    # Average row
    avg_row = {'Trait': 'Average'}
    for cfg_name in ABLATION_CONFIGS:
        all_means = []
        for trait in TRAITS:
            if trait in all_results and cfg_name in all_results[trait]:
                runs = all_results[trait][cfg_name]
                if runs:
                    all_means.append(np.mean([r['pcc'] for r in runs]))
        avg_row[cfg_name] = f"{np.mean(all_means):.4f}" if all_means else '-'
    rows.append(avg_row)

    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / 'ablation_table.csv', index=False, encoding='utf-8-sig')

    logger.info("\n" + "=" * 60)
    logger.info("ABLATION RESULTS")
    logger.info("=" * 60)
    print(df.to_string(index=False))
    logger.info(f"\n结果保存: {RESULT_DIR}")


if __name__ == '__main__':
    main()

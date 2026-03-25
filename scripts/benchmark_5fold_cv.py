#!/usr/bin/env python3
"""
PlantHGNN 完整基准测试 - 5折交叉验证

比较以下方法：
  1. GBLUP          - 统计学基线
  2. Ridge           - 线性机器学习基线
  3. DNNGP          - 深度神经网络基线
  4. NetGP          - 图网络基线（直接竞争对手）
  5. Transformer     - 纯Transformer（消融：无GCN无AttnRes）
  6. Transformer+AttnRes  - 消融：无GCN
  7. Transformer+GCN      - 消融：无AttnRes
  8. PlantHGNN_Full       - 完整模型

评估指标：PCC (mean ± std), Spearman r, NDCG@10
统计显著性：Wilcoxon signed-rank test (vs best baseline = NetGP)

输出：
  results/gstp007/benchmark_5fold_cv.json    — 完整结果
  results/gstp007/benchmark_table.csv        — 论文用表格
  results/gstp007/benchmark_per_trait.csv    — 逐性状结果
"""

import sys
import os
sys.path.insert(0, 'E:/GWAS')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
from pathlib import Path
import json
import time
import logging
import warnings
warnings.filterwarnings('ignore')

# ── Setup logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('E:/GWAS/results/gstp007/benchmark_5fold_cv.log',
                            mode='w', encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)

# ── 导入模块 ───────────────────────────────────────────────────────────────────
from src.models.plant_hgnn import PlantHGNN
from src.models.baselines.gblup import GBLUP
from src.models.baselines.dnngp import DNNGP
from src.models.baselines.netgp import NetGP
from src.training.metrics import (
    compute_pearson_correlation,
    compute_spearman_correlation,
    compute_ndcg,
    wilcoxon_test,
    format_metric_with_significance,
)

# ── 路径 ───────────────────────────────────────────────────────────────────────
PROC_DIR   = Path('E:/GWAS/data/processed/gstp007')
GRAPH_DIR  = PROC_DIR / 'graph'
RESULT_DIR = Path('E:/GWAS/results/gstp007')
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Device: {DEVICE}")

TRAITS = [
    'Plant_Height', 'Grain_Length', 'Grain_Width',
    'Days_to_Heading', 'Panicle_Length', 'Grain_Weight', 'Yield_per_plant',
]

# ── 超参数 ─────────────────────────────────────────────────────────────────────
N_FOLDS   = 5
SEEDS     = [42, 123, 456]   # 3 seeds × 5 folds = 15 runs per model per trait
TOP_K_SNP = 5000             # PCS top-K

CFG = dict(
    batch_size   = 64,
    lr           = 5e-4,
    weight_decay = 1e-4,
    max_epochs   = 150,
    patience     = 20,
    d_model      = 128,
    n_layers     = 6,
    n_blocks     = 8,
    n_heads      = 8,
    dropout      = 0.2,
)


# ══════════════════════════════════════════════════════════════════════════════
# PCS 特征选择（训练集内，防止数据泄漏）
# ══════════════════════════════════════════════════════════════════════════════

def pcs_select(X_train: np.ndarray, y_train: np.ndarray,
               top_k: int = TOP_K_SNP) -> np.ndarray:
    """在训练集上选top-K个Pearson |r| SNP，返回索引数组。"""
    y_c = y_train - y_train.mean()
    y_norm = np.sqrt((y_c ** 2).sum()) + 1e-8
    X_c = X_train - X_train.mean(axis=0)
    r = (X_c * y_c[:, None]).sum(axis=0) / \
        (np.sqrt((X_c ** 2).sum(axis=0)) + 1e-8) / y_norm
    k = min(top_k, X_train.shape[1])
    idx = np.argpartition(np.abs(r), -k)[-k:]
    return idx


# ══════════════════════════════════════════════════════════════════════════════
# PlantHGNN PyTorch 训练器
# ══════════════════════════════════════════════════════════════════════════════

class GPDataset(torch.utils.data.Dataset):
    def __init__(self, X_snp, y, X_gene=None):
        self.X_snp  = torch.tensor(X_snp,  dtype=torch.float32)
        self.y      = torch.tensor(y,       dtype=torch.float32)
        self.X_gene = torch.tensor(X_gene, dtype=torch.float32) \
                      if X_gene is not None else None

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        if self.X_gene is not None:
            return self.X_snp[i], self.y[i], self.X_gene[i]
        return self.X_snp[i], self.y[i]


def train_planthgnn(
    X_tr, y_tr, X_va, y_va,
    n_snps: int,
    gene_tr=None, gene_va=None,
    adj=None,
    use_gcn=True, use_attnres=True,
    cfg=CFG,
    desc="PlantHGNN",
):
    """训练PlantHGNN变体，返回 (val_pcc, model)"""
    n_genes = gene_tr.shape[1] if (gene_tr is not None and use_gcn) else 0
    model = PlantHGNN(
        n_snps          = n_snps,
        d_model         = cfg['d_model'],
        n_transformer_layers = cfg['n_layers'],
        n_attnres_blocks = cfg['n_blocks'],
        n_traits        = 1,
        n_gcn_genes     = n_genes,
        n_views         = 1,
        use_gcn         = use_gcn and (n_genes > 0),
        use_attnres     = use_attnres,
        n_heads         = cfg['n_heads'],
        dropout         = cfg['dropout'],
    ).to(DEVICE)

    opt = optim.AdamW(model.parameters(),
                      lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg['max_epochs'], eta_min=1e-5
    )

    # Prepare adj
    adj_list_dev = None
    if use_gcn and adj is not None and n_genes > 0:
        adj_t = torch.tensor(adj, dtype=torch.float32).to(DEVICE)
        adj_list_dev = [adj_t]

    # Dataset
    use_gene = (use_gcn and gene_tr is not None and n_genes > 0)
    ds_tr = GPDataset(X_tr, y_tr, gene_tr if use_gene else None)
    ds_va = GPDataset(X_va, y_va, gene_va if use_gene else None)
    dl_tr = DataLoader(ds_tr, batch_size=cfg['batch_size'], shuffle=True)

    best_pcc   = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(cfg['max_epochs']):
        model.train()
        for batch in dl_tr:
            if len(batch) == 3:
                xb, yb, gb = [t.to(DEVICE) for t in batch]
            else:
                xb, yb = [t.to(DEVICE) for t in batch]
                gb = None

            pred = model(
                xb,
                gene_feat=gb,
                adj_list=adj_list_dev,
            )
            loss = nn.MSELoss()(pred, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        # Validation
        model.eval()
        all_pred = []
        with torch.no_grad():
            for batch in DataLoader(ds_va, batch_size=256):
                if len(batch) == 3:
                    xb, _, gb = [t.to(DEVICE) for t in batch]
                else:
                    xb = batch[0].to(DEVICE)
                    gb = None
                p = model(xb, gene_feat=gb, adj_list=adj_list_dev)
                all_pred.append(p.cpu().numpy())
        preds = np.concatenate(all_pred)
        pcc, _ = pearsonr(y_va, preds)

        if pcc > best_pcc:
            best_pcc   = pcc
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg['patience']:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_pcc, model


def predict_planthgnn(model, X_te, gene_te=None, adj=None, use_gcn=True):
    model.eval()
    adj_list_dev = None
    if use_gcn and adj is not None:
        adj_t = torch.tensor(adj, dtype=torch.float32).to(DEVICE)
        adj_list_dev = [adj_t]

    use_gene = (use_gcn and gene_te is not None)
    ds = GPDataset(X_te, np.zeros(len(X_te)), gene_te if use_gene else None)
    all_pred = []
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=256):
            if len(batch) == 3:
                xb, _, gb = [t.to(DEVICE) for t in batch]
            else:
                xb = batch[0].to(DEVICE)
                gb = None
            p = model(xb, gene_feat=gb, adj_list=adj_list_dev)
            all_pred.append(p.cpu().numpy())
    return np.concatenate(all_pred)


# ══════════════════════════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════════════════════════

def load_trait_data(trait: str):
    """
    加载性状数据（合并train/val/test为全量数据，供5折CV使用）。
    返回: X_all (n, n_snps), y_all (n,), gene_all (n, n_genes) or None, adj or None
    """
    td = PROC_DIR / trait
    X_tr = np.load(td / 'X_train.npy')
    X_va = np.load(td / 'X_val.npy')
    X_te = np.load(td / 'X_test.npy')
    y_tr = np.load(td / 'y_train.npy')      # 原始BLUP值（未标准化）
    y_va = np.load(td / 'y_val.npy')
    y_te = np.load(td / 'y_test.npy')

    X_all = np.concatenate([X_tr, X_va, X_te], axis=0)
    y_all = np.concatenate([y_tr, y_va, y_te], axis=0)

    # GCN 基因特征（如果存在）
    gene_all = None
    adj = None
    gene_files = list((GRAPH_DIR / trait).glob('gene_feat_*.npy')) \
                 if (GRAPH_DIR / trait).exists() else []
    if gene_files:
        # 合并 train/val/test gene features
        gf_tr = np.load(GRAPH_DIR / trait / 'gene_feat_train.npy') \
                if (GRAPH_DIR / trait / 'gene_feat_train.npy').exists() else None
        gf_va = np.load(GRAPH_DIR / trait / 'gene_feat_val.npy') \
                if (GRAPH_DIR / trait / 'gene_feat_val.npy').exists() else None
        gf_te = np.load(GRAPH_DIR / trait / 'gene_feat_test.npy') \
                if (GRAPH_DIR / trait / 'gene_feat_test.npy').exists() else None
        if gf_tr is not None:
            gene_all = np.concatenate([gf_tr, gf_va, gf_te], axis=0)

    for adj_path in [GRAPH_DIR / 'adj_norm.npy', GRAPH_DIR / 'adj_norm.npz',
                     GRAPH_DIR / 'ppi_adj.npz', GRAPH_DIR / 'ppi_adj.npy']:
        if adj_path.exists():
            if adj_path.suffix == '.npy':
                adj = np.load(str(adj_path))
            else:
                adj = sp.load_npz(str(adj_path)).toarray().astype(np.float32)
            break

    return X_all, y_all, gene_all, adj


# ══════════════════════════════════════════════════════════════════════════════
# 单折实验
# ══════════════════════════════════════════════════════════════════════════════

def run_fold(
    trait: str,
    X_all: np.ndarray,
    y_all: np.ndarray,
    gene_all,
    adj,
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    fold: int,
    seed: int,
    results: dict,
):
    """在单个折上运行所有方法并记录结果。"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 80% train → 12.5% val
    n_val = max(int(len(tr_idx) * 0.15), 1)
    rng   = np.random.RandomState(seed)
    va_local = rng.choice(len(tr_idx), size=n_val, replace=False)
    tr_local = np.setdiff1d(np.arange(len(tr_idx)), va_local)
    actual_tr_idx = tr_idx[tr_local]
    actual_va_idx = tr_idx[va_local]

    X_tr_raw = X_all[actual_tr_idx]
    X_va_raw = X_all[actual_va_idx]
    X_te_raw = X_all[te_idx]
    y_tr = y_all[actual_tr_idx]
    y_va = y_all[actual_va_idx]
    y_te = y_all[te_idx]

    # ── PCS 特征选择（仅在训练集上）──────────────────────────────────────────
    snp_idx = pcs_select(X_tr_raw, y_tr, top_k=TOP_K_SNP)
    X_tr = X_tr_raw[:, snp_idx]
    X_va = X_va_raw[:, snp_idx]
    X_te = X_te_raw[:, snp_idx]
    n_snps = X_tr.shape[1]

    # ── 标准化（基于训练集）──────────────────────────────────────────────────
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_va   = scaler.transform(X_va)
    X_te   = scaler.transform(X_te)

    y_mean = y_tr.mean()
    y_std  = y_tr.std() + 1e-8
    y_tr_s = (y_tr - y_mean) / y_std
    y_va_s = (y_va - y_mean) / y_std
    # 预测时转回原始尺度再算PCC

    # ── 基因特征 ─────────────────────────────────────────────────────────────
    if gene_all is not None:
        g_tr = gene_all[actual_tr_idx]
        g_va = gene_all[actual_va_idx]
        g_te = gene_all[te_idx]
        # 标准化gene features
        g_scaler = StandardScaler()
        g_flat_tr = g_tr.reshape(-1, 1)
        g_scaler.fit(g_flat_tr)
        g_tr = g_scaler.transform(g_tr.reshape(-1, 1)).reshape(g_tr.shape)
        g_va = g_scaler.transform(g_va.reshape(-1, 1)).reshape(g_va.shape)
        g_te = g_scaler.transform(g_te.reshape(-1, 1)).reshape(g_te.shape)
    else:
        g_tr = g_va = g_te = None

    fold_key = f"fold{fold}_seed{seed}"
    logger.info(f"  [{trait}] {fold_key}: tr={len(X_tr)} va={len(X_va)} te={len(X_te)} snps={n_snps}")

    def record(model_name, y_pred):
        pcc, _   = pearsonr(y_te, y_pred)
        scc, _   = spearmanr(y_te, y_pred)
        if trait not in results:
            results[trait] = {}
        if model_name not in results[trait]:
            results[trait][model_name] = []
        results[trait][model_name].append({
            'fold': fold, 'seed': seed,
            'pcc': float(pcc), 'spearman': float(scc),
        })
        return pcc

    # ── 1. GBLUP ─────────────────────────────────────────────────────────────
    try:
        gblup = GBLUP()
        gblup.fit(X_tr, y_tr_s)
        pred = gblup.predict(X_te) * y_std + y_mean
        pcc = record('GBLUP', pred)
        logger.info(f"    GBLUP      PCC={pcc:.4f}")
    except Exception as e:
        logger.warning(f"    GBLUP failed: {e}")

    # ── 2. Ridge ──────────────────────────────────────────────────────────────
    try:
        ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
        ridge.fit(X_tr, y_tr_s)
        pred = ridge.predict(X_te) * y_std + y_mean
        pcc = record('Ridge', pred)
        logger.info(f"    Ridge      PCC={pcc:.4f}")
    except Exception as e:
        logger.warning(f"    Ridge failed: {e}")

    # ── 3. DNNGP ──────────────────────────────────────────────────────────────
    try:
        dnngp = DNNGP(
            hidden_dims=(512, 256, 128, 64),
            dropout=0.3, lr=1e-3,
            max_epochs=CFG['max_epochs'], patience=CFG['patience'],
        )
        dnngp.fit(X_tr, y_tr_s, X_val=X_va, y_val=y_va_s)
        pred = dnngp.predict(X_te) * y_std + y_mean
        pcc = record('DNNGP', pred)
        logger.info(f"    DNNGP      PCC={pcc:.4f}")
    except Exception as e:
        logger.warning(f"    DNNGP failed: {e}")

    # ── 4. NetGP ──────────────────────────────────────────────────────────────
    try:
        netgp = NetGP(
            d_hidden=128, dropout=0.2, lr=5e-4,
            max_epochs=CFG['max_epochs'], patience=CFG['patience'],
        )
        netgp.fit(X_tr, y_tr_s, X_val=X_va, y_val=y_va_s,
                  gene_train=g_tr, gene_val=g_va, adj=adj)
        pred = netgp.predict(X_te, gene_test=g_te) * y_std + y_mean
        pcc = record('NetGP', pred)
        logger.info(f"    NetGP      PCC={pcc:.4f}")
    except Exception as e:
        logger.warning(f"    NetGP failed: {e}")

    # ── 5. Transformer Only ───────────────────────────────────────────────────
    try:
        _, m = train_planthgnn(
            X_tr, y_tr_s, X_va, y_va_s, n_snps,
            use_gcn=False, use_attnres=False, cfg=CFG,
        )
        pred = predict_planthgnn(m, X_te) * y_std + y_mean
        pcc = record('Transformer', pred)
        logger.info(f"    Transformer PCC={pcc:.4f}")
        del m; torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(f"    Transformer failed: {e}")

    # ── 6. Transformer + AttnRes ──────────────────────────────────────────────
    try:
        _, m = train_planthgnn(
            X_tr, y_tr_s, X_va, y_va_s, n_snps,
            use_gcn=False, use_attnres=True, cfg=CFG,
        )
        pred = predict_planthgnn(m, X_te) * y_std + y_mean
        pcc = record('Transformer+AttnRes', pred)
        logger.info(f"    Tr+AttnRes  PCC={pcc:.4f}")
        del m; torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(f"    Tr+AttnRes failed: {e}")

    # ── 7. Transformer + GCN (no AttnRes) ────────────────────────────────────
    if g_tr is not None and adj is not None:
        try:
            _, m = train_planthgnn(
                X_tr, y_tr_s, X_va, y_va_s, n_snps,
                gene_tr=g_tr, gene_va=g_va,
                adj=adj,
                use_gcn=True, use_attnres=False, cfg=CFG,
            )
            pred = predict_planthgnn(m, X_te, g_te, adj, use_gcn=True) * y_std + y_mean
            pcc = record('Transformer+GCN', pred)
            logger.info(f"    Tr+GCN      PCC={pcc:.4f}")
            del m; torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"    Tr+GCN failed: {e}")

    # ── 8. PlantHGNN Full ─────────────────────────────────────────────────────
    if g_tr is not None and adj is not None:
        try:
            _, m = train_planthgnn(
                X_tr, y_tr_s, X_va, y_va_s, n_snps,
                gene_tr=g_tr, gene_va=g_va,
                adj=adj,
                use_gcn=True, use_attnres=True, cfg=CFG,
            )
            pred = predict_planthgnn(m, X_te, g_te, adj, use_gcn=True) * y_std + y_mean
            pcc = record('PlantHGNN_Full', pred)
            logger.info(f"    PlantHGNN   PCC={pcc:.4f}")
            del m; torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"    PlantHGNN failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 汇总统计
# ══════════════════════════════════════════════════════════════════════════════

def summarize_results(results: dict) -> dict:
    """计算各方法在各性状上的 mean ± std PCC。"""
    summary = {}
    for trait, models in results.items():
        summary[trait] = {}
        for model_name, runs in models.items():
            pccs = [r['pcc'] for r in runs]
            spcs = [r['spearman'] for r in runs]
            summary[trait][model_name] = {
                'pcc_mean': float(np.mean(pccs)),
                'pcc_std':  float(np.std(pccs)),
                'spearman_mean': float(np.mean(spcs)),
                'spearman_std':  float(np.std(spcs)),
                'n_runs': len(runs),
            }
    return summary


def compute_significance(results: dict, baseline: str = 'NetGP') -> dict:
    """Wilcoxon检验：各方法 vs baseline。"""
    sig = {}
    for trait, models in results.items():
        sig[trait] = {}
        if baseline not in models:
            continue
        base_pccs = [r['pcc'] for r in models[baseline]]
        for model_name, runs in models.items():
            if model_name == baseline:
                sig[trait][model_name] = 1.0
                continue
            pccs = [r['pcc'] for r in runs]
            # Align lengths
            n = min(len(pccs), len(base_pccs))
            p = wilcoxon_test(pccs[:n], base_pccs[:n])
            sig[trait][model_name] = float(p)
    return sig


def build_paper_table(summary: dict, sig: dict, baseline: str = 'NetGP') -> pd.DataFrame:
    """生成论文格式表格（性状 × 方法，PCC mean ± std *）"""
    traits  = list(summary.keys())
    models  = []
    for t in traits:
        for m in summary[t].keys():
            if m not in models:
                models.append(m)

    rows = []
    for trait in traits:
        row = {'Trait': trait}
        for model in models:
            if model not in summary[trait]:
                row[model] = '-'
                continue
            s = summary[trait][model]
            p = sig.get(trait, {}).get(model, 1.0)
            if p < 0.001:   stars = '***'
            elif p < 0.01:  stars = '**'
            elif p < 0.05:  stars = '*'
            else:           stars = ''
            row[model] = f"{s['pcc_mean']:.4f}±{s['pcc_std']:.4f}{stars}"
        rows.append(row)

    # 添加平均行
    avg_row = {'Trait': 'Average'}
    for model in models:
        means = []
        for t in traits:
            if model in summary[t]:
                means.append(summary[t][model]['pcc_mean'])
        if means:
            avg_row[model] = f"{np.mean(means):.4f}"
        else:
            avg_row[model] = '-'
    rows.append(avg_row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 70)
    logger.info("PlantHGNN 5-Fold CV Benchmark")
    logger.info("=" * 70)

    all_results = {}
    start_total = time.time()

    for trait in TRAITS:
        logger.info(f"\n{'='*60}")
        logger.info(f"性状: {trait}")
        logger.info(f"{'='*60}")

        try:
            X_all, y_all, gene_all, adj = load_trait_data(trait)
        except Exception as e:
            logger.warning(f"  加载数据失败: {e}, 跳过")
            continue

        logger.info(f"  数据: X={X_all.shape}, y={y_all.shape}, "
                    f"gene={'yes' if gene_all is not None else 'no'}, "
                    f"adj={'yes' if adj is not None else 'no'}")

        for seed in SEEDS:
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            for fold, (tr_idx, te_idx) in enumerate(kf.split(X_all)):
                try:
                    run_fold(trait, X_all, y_all, gene_all, adj,
                             tr_idx, te_idx, fold, seed, all_results)
                except Exception as e:
                    logger.error(f"  Fold {fold} seed {seed} failed: {e}")

    # ── 保存原始结果 ──────────────────────────────────────────────────────────
    raw_path = RESULT_DIR / 'benchmark_5fold_cv.json'
    with open(raw_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\n原始结果已保存: {raw_path}")

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    summary = summarize_results(all_results)
    sig     = compute_significance(all_results, baseline='NetGP')

    summary_path = RESULT_DIR / 'benchmark_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({'summary': summary, 'significance_vs_NetGP': sig}, f, indent=2)

    # ── 论文表格 ──────────────────────────────────────────────────────────────
    table = build_paper_table(summary, sig)
    table_path = RESULT_DIR / 'benchmark_table.csv'
    table.to_csv(table_path, index=False, encoding='utf-8-sig')
    logger.info(f"论文表格已保存: {table_path}")

    # ── 打印汇总 ──────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("BENCHMARK RESULTS (PCC mean ± std)")
    logger.info("=" * 70)

    # 打印 ASCII 表格
    all_models = []
    for t in TRAITS:
        if t in summary:
            for m in summary[t]:
                if m not in all_models:
                    all_models.append(m)

    header = f"{'Trait':<22}" + "".join(f"{m[:14]:>17}" for m in all_models)
    logger.info(header)
    logger.info("-" * len(header))

    for trait in TRAITS:
        if trait not in summary:
            continue
        row = f"{trait:<22}"
        for m in all_models:
            if m in summary[trait]:
                s = summary[trait][m]
                row += f"  {s['pcc_mean']:.4f}±{s['pcc_std']:.4f}"
            else:
                row += "               -"
        logger.info(row)

    # 平均行
    logger.info("-" * len(header))
    avg_row = f"{'Average':<22}"
    for m in all_models:
        means = [summary[t][m]['pcc_mean']
                 for t in TRAITS if t in summary and m in summary[t]]
        avg_row += f"  {np.mean(means):.4f}        " if means else "               -"
    logger.info(avg_row)

    elapsed = time.time() - start_total
    logger.info(f"\n总耗时: {elapsed/60:.1f} 分钟")
    logger.info(f"结果目录: {RESULT_DIR}")

    print(table.to_string(index=False))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
PlantHGNN 可解释性分析

分析模块：
  1. 网络贡献度分析 (Network Contribution)
     - 多视图GCN的注意力权重 α = [α_ppi, α_go, α_pathway]
     - 热力图：性状 × 网络类型

  2. AttnRes 深度注意力分析
     - block × block 注意力权重矩阵
     - 分析"最终表示主要来自第几层"

  3. SNP 重要性分析（梯度法）
     - 对每个性状计算每个SNP的梯度范数
     - Manhattan-style 散点图

  4. 基因嵌入 UMAP（可选，需要GCN）
     - 学到的基因嵌入降维可视化

输出：
  results/gstp007/figures/
  results/gstp007/interpretability/
"""

import sys
sys.path.insert(0, 'E:/GWAS')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
from pathlib import Path
import json
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

from src.models.plant_hgnn import PlantHGNN

PROC_DIR   = Path('E:/GWAS/data/processed/gstp007')
GRAPH_DIR  = PROC_DIR / 'graph'
RESULT_DIR = Path('E:/GWAS/results/gstp007')
FIG_DIR    = RESULT_DIR / 'figures'
INTERP_DIR = RESULT_DIR / 'interpretability'
FIG_DIR.mkdir(parents=True, exist_ok=True)
INTERP_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAITS = [
    'Plant_Height', 'Grain_Length', 'Grain_Width',
    'Days_to_Heading', 'Panicle_Length', 'Grain_Weight', 'Yield_per_plant',
]

CFG = dict(
    batch_size=64, lr=5e-4, weight_decay=1e-4,
    max_epochs=150, patience=20,
    d_model=128, n_layers=6, n_blocks=8, n_heads=8, dropout=0.2,
)


# ══════════════════════════════════════════════════════════════════════════════
# 训练最终模型（用全量train数据）
# ══════════════════════════════════════════════════════════════════════════════

def pcs_select(X_tr, y_tr, top_k=5000):
    y_c  = y_tr - y_tr.mean()
    y_n  = np.sqrt((y_c**2).sum()) + 1e-8
    X_c  = X_tr - X_tr.mean(0)
    r    = (X_c * y_c[:, None]).sum(0) / (np.sqrt((X_c**2).sum(0)) + 1e-8) / y_n
    k    = min(top_k, X_tr.shape[1])
    return np.argpartition(np.abs(r), -k)[-k:]


def load_trait(trait):
    td = PROC_DIR / trait
    X_tr = np.load(td / 'X_train.npy'); y_tr = np.load(td / 'y_train.npy')
    X_va = np.load(td / 'X_val.npy');   y_va = np.load(td / 'y_val.npy')
    X_te = np.load(td / 'X_test.npy');  y_te = np.load(td / 'y_test.npy')

    gene = {}
    adj = None
    gd = GRAPH_DIR / trait
    for split in ['train', 'val', 'test']:
        p = gd / f'gene_feat_{split}.npy' if gd.exists() else Path('__none__')
        gene[split] = np.load(str(p)) if p.exists() else None

    for ap in [GRAPH_DIR / 'adj_norm.npy', GRAPH_DIR / 'adj_norm.npz',
               GRAPH_DIR / 'ppi_adj.npz', GRAPH_DIR / 'ppi_adj.npy']:
        if ap.exists():
            adj = np.load(str(ap)) if ap.suffix == '.npy' \
                  else sp.load_npz(str(ap)).toarray().astype(np.float32)
            break

    return (X_tr, y_tr, X_va, y_va, X_te, y_te,
            gene['train'], gene['val'], gene['test'], adj)


def train_final_model(trait):
    """训练完整PlantHGNN模型用于可解释性分析。"""
    (X_tr, y_tr, X_va, y_va, X_te, y_te,
     g_tr, g_va, g_te, adj) = load_trait(trait)

    # PCS + 标准化
    snp_idx = pcs_select(X_tr, y_tr)
    scl = StandardScaler()
    X_tr = scl.fit_transform(X_tr[:, snp_idx])
    X_va = scl.transform(X_va[:, snp_idx])
    X_te = scl.transform(X_te[:, snp_idx])
    n_snps = X_tr.shape[1]

    ym, ys = y_tr.mean(), y_tr.std() + 1e-8
    y_tr_s = (y_tr - ym) / ys
    y_va_s = (y_va - ym) / ys

    use_gcn = (g_tr is not None and adj is not None)
    if use_gcn:
        gs  = StandardScaler()
        g_tr = gs.fit_transform(g_tr)
        g_va = gs.transform(g_va)
        g_te = gs.transform(g_te)
        n_genes = g_tr.shape[1]
        adj_dev = [torch.tensor(adj, dtype=torch.float32).to(DEVICE)]
    else:
        n_genes = 0
        adj_dev = None

    model = PlantHGNN(
        n_snps=n_snps, d_model=CFG['d_model'],
        n_transformer_layers=CFG['n_layers'],
        n_attnres_blocks=CFG['n_blocks'], n_traits=1,
        n_gcn_genes=n_genes, n_views=1,
        use_gcn=use_gcn, use_attnres=True,
        n_heads=CFG['n_heads'], dropout=CFG['dropout'],
    ).to(DEVICE)

    opt = optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG['max_epochs'], eta_min=1e-5)

    class GPDataset(torch.utils.data.Dataset):
        def __init__(self, X, y, G=None):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
            self.G = torch.tensor(G, dtype=torch.float32) if G is not None else None
        def __len__(self): return len(self.y)
        def __getitem__(self, i):
            if self.G is not None: return self.X[i], self.y[i], self.G[i]
            return self.X[i], self.y[i]

    dl = DataLoader(
        GPDataset(X_tr, y_tr_s, g_tr if use_gcn else None),
        batch_size=CFG['batch_size'], shuffle=True,
    )

    best_pcc, best_state, no_imp = -1.0, None, 0
    for epoch in range(CFG['max_epochs']):
        model.train()
        for batch in dl:
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
        Gv = torch.tensor(g_va, dtype=torch.float32).to(DEVICE) if use_gcn else None
        with torch.no_grad():
            pv = model(Xv, gene_feat=Gv, adj_list=adj_dev).cpu().numpy()
        pcc = pearsonr(y_va_s, pv)[0]
        if pcc > best_pcc:
            best_pcc = pcc; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= CFG['patience']: break

    if best_state: model.load_state_dict(best_state)

    # Test PCC
    model.eval()
    Xt = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
    Gt = torch.tensor(g_te, dtype=torch.float32).to(DEVICE) if use_gcn else None
    with torch.no_grad():
        pt = model(Xt, gene_feat=Gt, adj_list=adj_dev).cpu().numpy()
    test_pcc = pearsonr(y_te, pt * ys + ym)[0]
    logger.info(f"  [{trait}] Final model: val_PCC={best_pcc:.4f}, test_PCC={test_pcc:.4f}")

    return model, {
        'X_te': X_te, 'y_te': y_te, 'g_te': g_te,
        'adj_dev': adj_dev, 'adj': adj,
        'snp_idx': snp_idx, 'use_gcn': use_gcn,
        'ym': ym, 'ys': ys,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 分析 1: AttnRes 深度注意力权重
# ══════════════════════════════════════════════════════════════════════════════

def analyze_depth_attention(model, X_sample, G_sample, adj_dev, trait_name):
    """提取AttnRes的block间注意力权重矩阵。"""
    model.eval()
    Xs = torch.tensor(X_sample[:32], dtype=torch.float32).to(DEVICE)
    Gs = torch.tensor(G_sample[:32], dtype=torch.float32).to(DEVICE) \
         if G_sample is not None else None

    with torch.no_grad():
        _ = model(Xs, gene_feat=Gs, adj_list=adj_dev)

    weights = model.get_depth_attention_weights()
    if weights is None:
        logger.info(f"  [{trait_name}] No AttnRes weights available")
        return None

    w = weights.detach().cpu().numpy()   # (n_blocks, n_reg)
    np.save(INTERP_DIR / f'{trait_name}_depth_attn.npy', w)
    logger.info(f"  [{trait_name}] Depth attention shape: {w.shape}")
    return w


# ══════════════════════════════════════════════════════════════════════════════
# 分析 2: SNP 梯度重要性
# ══════════════════════════════════════════════════════════════════════════════

def compute_snp_gradient_importance(model, X_te, G_te, adj_dev, snp_meta_path):
    """
    用输入梯度计算每个SNP对预测的重要性。
    返回: snp_importance (n_snps,)
    """
    model.eval()
    Xt = torch.tensor(X_te, dtype=torch.float32).to(DEVICE).requires_grad_(True)
    Gt = torch.tensor(G_te, dtype=torch.float32).to(DEVICE) if G_te is not None else None

    pred = model(Xt, gene_feat=Gt, adj_list=adj_dev)
    pred.sum().backward()

    grad = Xt.grad.detach().cpu().numpy()  # (n_te, n_snps)
    importance = np.abs(grad).mean(axis=0) # (n_snps,)
    return importance


# ══════════════════════════════════════════════════════════════════════════════
# 可视化（带 matplotlib）
# ══════════════════════════════════════════════════════════════════════════════

def plot_depth_attention_heatmap(depth_weights_dict):
    """绘制AttnRes深度注意力热力图（多性状）。"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import seaborn as sns

        n_traits = len(depth_weights_dict)
        if n_traits == 0: return

        fig, axes = plt.subplots(1, n_traits, figsize=(3 * n_traits, 4))
        if n_traits == 1: axes = [axes]

        for ax, (trait, W) in zip(axes, depth_weights_dict.items()):
            if W is None: continue
            sns.heatmap(W, ax=ax, cmap='YlOrRd', vmin=0,
                        annot=False, xticklabels=False, yticklabels=False)
            ax.set_title(trait.replace('_', '\n'), fontsize=8)
            ax.set_xlabel('Block (key)', fontsize=7)
            ax.set_ylabel('Block (query)', fontsize=7)

        plt.suptitle('AttnRes: Block-to-Block Attention Weights', fontsize=11)
        plt.tight_layout()
        save_path = FIG_DIR / 'depth_attention_heatmap.pdf'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {save_path}")
    except ImportError:
        logger.warning("matplotlib/seaborn not available, skip plot")


def plot_snp_importance_manhattan(importance_dict, snp_meta_dict):
    """绘制SNP重要性曼哈顿图。"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        n = len(importance_dict)
        if n == 0: return
        fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n))
        if n == 1: axes = [axes]

        chroms = list(range(1, 13))   # Rice has 12 chromosomes
        chrom_colors = {c: cm.tab20(i / 12) for i, c in enumerate(chroms)}

        for ax, (trait, importance) in zip(axes, importance_dict.items()):
            meta = snp_meta_dict.get(trait)
            if meta is None or len(importance) == 0:
                continue

            n_snps = min(len(importance), len(meta))
            imp = importance[:n_snps]
            chrs = meta['chr'].values[:n_snps].astype(int) if 'chr' in meta.columns else np.ones(n_snps, dtype=int)
            pos  = np.arange(n_snps)

            colors = [chrom_colors.get(c % 12 + 1, 'gray') for c in chrs]
            ax.bar(pos, imp, color=colors, width=1.0, linewidth=0)
            ax.set_xlim(-1, n_snps + 1)
            ax.set_ylabel('|Gradient|', fontsize=8)
            ax.set_title(trait.replace('_', ' '), fontsize=9)
            ax.set_xlabel('SNP (sorted by chr:pos)', fontsize=7)
            ax.tick_params(axis='x', labelsize=6)

        plt.suptitle('SNP Importance (Input Gradient)', y=1.02, fontsize=12)
        plt.tight_layout()
        save_path = FIG_DIR / 'snp_importance_manhattan.pdf'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {save_path}")
    except ImportError:
        logger.warning("matplotlib not available, skip plot")


def plot_summary_bar(summary_dict):
    """绘制各方法平均PCC条形图。"""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        models = list(summary_dict.keys())
        means  = [summary_dict[m]['mean'] for m in models]
        stds   = [summary_dict[m]['std']  for m in models]
        colors = ['#4472C4' if 'PlantHGNN' not in m else '#ED7D31' for m in models]

        bars = ax.bar(range(len(models)), means, yerr=stds,
                      color=colors, capsize=4, width=0.6)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace('_', '\n') for m in models],
                           fontsize=8, rotation=30, ha='right')
        ax.set_ylabel('Average PCC (7 traits)', fontsize=10)
        ax.set_title('PlantHGNN vs Baselines (5-fold CV)', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, min(1.0, max(means) * 1.15))

        # Add value labels
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{m:.4f}', ha='center', va='bottom', fontsize=7)

        plt.tight_layout()
        save_path = FIG_DIR / 'model_comparison_bar.pdf'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {save_path}")
    except ImportError:
        logger.warning("matplotlib not available, skip bar plot")


# ══════════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("PlantHGNN Interpretability Analysis")
    logger.info("=" * 60)

    depth_weights_dict   = {}
    importance_dict      = {}
    snp_meta_dict        = {}

    for trait in TRAITS:
        logger.info(f"\n--- {trait} ---")
        try:
            model, data = train_final_model(trait)
        except Exception as e:
            logger.warning(f"  Training failed: {e}")
            continue

        X_te   = data['X_te']
        g_te   = data['g_te']
        adj_dev = data['adj_dev']
        snp_idx = data['snp_idx']

        # 1. Depth attention weights
        G_sample = g_te if data['use_gcn'] else None
        dw = analyze_depth_attention(model, X_te, G_sample, adj_dev, trait)
        depth_weights_dict[trait] = dw

        # 2. SNP gradient importance
        try:
            importance = compute_snp_gradient_importance(
                model, X_te[:64], G_sample[:64] if G_sample is not None else None,
                adj_dev,
                snp_meta_path=None,
            )
            importance_dict[trait] = importance
            np.save(INTERP_DIR / f'{trait}_snp_importance.npy', importance)

            # Load SNP meta for plotting
            meta_path = PROC_DIR / trait / 'selected_snp_meta.csv'
            if meta_path.exists():
                meta = pd.read_csv(meta_path)
                # Reindex by snp_idx to match the selected SNPs
                snp_meta_dict[trait] = meta
                # Save top-50 SNPs
                top50 = np.argsort(importance)[::-1][:50]
                top50_meta = meta.iloc[top50].copy()
                top50_meta['importance'] = importance[top50]
                top50_meta.to_csv(
                    INTERP_DIR / f'{trait}_top50_snps.csv', index=False
                )
                logger.info(f"  [{trait}] Top-5 SNPs: {meta.iloc[top50[:5]]['snp_id'].tolist()}")
        except Exception as e:
            logger.warning(f"  SNP importance failed: {e}")

        del model

    # ── Visualizations ────────────────────────────────────────────────────────
    logger.info("\nGenerating visualizations...")
    plot_depth_attention_heatmap(depth_weights_dict)
    plot_snp_importance_manhattan(importance_dict, snp_meta_dict)

    # ── Load benchmark results for bar chart (if available) ──────────────────
    bench_path = RESULT_DIR / 'benchmark_summary.json'
    if bench_path.exists():
        with open(bench_path) as f:
            bench = json.load(f)
        summary = bench.get('summary', {})
        # Compute per-model average across all traits
        model_avgs = {}
        for trait in TRAITS:
            if trait not in summary: continue
            for mname, stats in summary[trait].items():
                if mname not in model_avgs:
                    model_avgs[mname] = []
                model_avgs[mname].append(stats['pcc_mean'])
        model_summary = {m: {'mean': np.mean(v), 'std': np.std(v)}
                         for m, v in model_avgs.items()}
        plot_summary_bar(model_summary)

    # ── Save interpretability summary ─────────────────────────────────────────
    summary_out = {}
    for trait in TRAITS:
        entry = {}
        if trait in depth_weights_dict and depth_weights_dict[trait] is not None:
            dw = depth_weights_dict[trait]
            # Most attended block (mean across all query blocks)
            entry['most_attended_block'] = int(np.argmax(dw.mean(0)))
        if trait in importance_dict:
            imp = importance_dict[trait]
            entry['top5_snp_indices'] = np.argsort(imp)[::-1][:5].tolist()
        summary_out[trait] = entry

    with open(INTERP_DIR / 'interpretability_summary.json', 'w') as f:
        json.dump(summary_out, f, indent=2)
    logger.info(f"\n可解释性分析完成，结果保存至: {INTERP_DIR}")


if __name__ == '__main__':
    main()

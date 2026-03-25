#!/usr/bin/env python3
"""
为GSTP007构建GCN所需的基因网络

步骤：
  1. 合并各性状的selected SNP位置，映射到水稻基因（±20kb窗口）
  2. 构建基因-基因PPI邻接矩阵（STRING v12, score>700）
  3. 对每个trait+split：计算每个基因的SNP特征均值
  4. 保存图数据（稀疏邻接 + 基因特征矩阵）

输出：
  data/processed/gstp007/graph/
    ├── ppi_adj.npz           (n_genes, n_genes) 稀疏邻接矩阵
    ├── gene_list.txt         基因ID列表
    └── {trait}/
        ├── gene_feat_train.npy  (n_train, n_genes) 基因特征
        ├── gene_feat_val.npy
        └── gene_feat_test.npy
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR   = Path('E:/GWAS/data')
NET_DIR    = DATA_DIR / 'raw' / 'networks'
PROC_DIR   = DATA_DIR / 'processed' / 'gstp007'
GRAPH_DIR  = PROC_DIR / 'graph'
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

GENE_WINDOW = 20_000   # ±20kb SNP→gene映射窗口

TRAITS = [
    'Plant_Height', 'Grain_Length', 'Grain_Width',
    'Days_to_Heading', 'Panicle_Length', 'Grain_Weight', 'Yield_per_plant',
]


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: 收集所有selected SNP的位置信息
# ══════════════════════════════════════════════════════════════════════════════

def collect_all_snp_metadata() -> pd.DataFrame:
    """合并所有性状的selected SNP元信息（去重）。"""
    all_meta = []
    for trait in TRAITS:
        meta_path = PROC_DIR / trait / 'selected_snp_meta.csv'
        if not meta_path.exists():
            continue
        df = pd.read_csv(meta_path)
        all_meta.append(df[['snp_id', 'chr', 'pos']].copy())

    if not all_meta:
        raise FileNotFoundError("未找到selected_snp_meta.csv，请先运行preprocess_gstp007.py")

    combined = pd.concat(all_meta).drop_duplicates(subset=['snp_id'])
    logger.info(f"合并所有性状: {len(combined):,} 个唯一SNP位置")
    return combined.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: SNP → 基因映射
# ══════════════════════════════════════════════════════════════════════════════

def map_snps_to_genes(snp_meta: pd.DataFrame, gene_ann: pd.DataFrame,
                      window: int = GENE_WINDOW) -> pd.DataFrame:
    """
    将SNP按物理位置映射到最近基因（基因体±window bp内）。
    """
    cache_path = GRAPH_DIR / 'snp_to_gene.csv'
    if cache_path.exists():
        logger.info(f"加载缓存SNP→Gene: {cache_path}")
        return pd.read_csv(cache_path)

    logger.info(f"SNP→Gene映射: {len(snp_meta):,} SNPs, "
                f"{len(gene_ann):,} 基因, 窗口={window:,} bp")

    gene_ann = gene_ann.copy()
    gene_ann['win_start'] = gene_ann['start'] - window
    gene_ann['win_end']   = gene_ann['end']   + window
    gene_ann['center']    = (gene_ann['start'] + gene_ann['end']) / 2

    results = []
    for chr_num, snp_group in snp_meta.groupby('chr'):
        genes_chr = gene_ann[gene_ann['chr'] == chr_num]
        if len(genes_chr) == 0:
            continue

        win_start = genes_chr['win_start'].values
        win_end   = genes_chr['win_end'].values
        centers   = genes_chr['center'].values
        gene_ids  = genes_chr['gene_id'].values

        for _, snp_row in snp_group.iterrows():
            pos = snp_row['pos']
            mask = (win_start <= pos) & (pos <= win_end)
            match_idx = np.where(mask)[0]
            if len(match_idx) == 0:
                continue
            elif len(match_idx) == 1:
                assigned_gene = gene_ids[match_idx[0]]
            else:
                dists = np.abs(centers[match_idx] - pos)
                assigned_gene = gene_ids[match_idx[np.argmin(dists)]]
            results.append({
                'snp_id':  snp_row['snp_id'],
                'chr':     chr_num,
                'pos':     int(pos),
                'gene_id': assigned_gene,
            })

    df = pd.DataFrame(results)
    n_assigned = len(df)
    n_total    = len(snp_meta)
    logger.info(f"SNP→Gene: {n_assigned:,}/{n_total:,} SNPs映射成功 "
                f"({n_assigned/n_total*100:.1f}%), "
                f"覆盖基因: {df['gene_id'].nunique():,}")

    df.to_csv(cache_path, index=False)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: 构建PPI邻接矩阵
# ══════════════════════════════════════════════════════════════════════════════

def build_ppi_adjacency(snp_to_gene: pd.DataFrame,
                         ppi: pd.DataFrame,
                         aliases: pd.DataFrame) -> tuple[sp.csr_matrix, list]:
    """
    构建覆盖SNP的基因子集的PPI邻接矩阵。

    Returns:
        adj_norm: 归一化邻接矩阵 (n_genes, n_genes)
        gene_list: 基因ID列表（与adj的行/列对应）
    """
    cache_adj  = GRAPH_DIR / 'ppi_adj.npz'
    cache_gene = GRAPH_DIR / 'gene_list.txt'

    if cache_adj.exists() and cache_gene.exists():
        logger.info("加载缓存PPI邻接矩阵...")
        adj = sp.load_npz(cache_adj)
        gene_list = cache_gene.read_text().strip().split('\n')
        logger.info(f"PPI邻接: {adj.shape}, 基因数: {len(gene_list)}")
        return adj, gene_list

    logger.info("构建PPI邻接矩阵...")

    # 所有被SNP覆盖的基因
    covered_genes = set(snp_to_gene['gene_id'].unique())
    logger.info(f"SNP覆盖基因数: {len(covered_genes)}")

    # 将protein_id映射到gene_id
    prot2gene = dict(zip(aliases['protein_id'], aliases['gene_id']))
    ppi = ppi.copy()
    ppi['gene1'] = ppi['protein1'].map(prot2gene)
    ppi['gene2'] = ppi['protein2'].map(prot2gene)
    ppi = ppi.dropna(subset=['gene1', 'gene2'])

    # 只保留两端都在covered_genes中的边
    mask = ppi['gene1'].isin(covered_genes) & ppi['gene2'].isin(covered_genes)
    ppi_sub = ppi[mask].copy()
    logger.info(f"PPI边数（covered基因间）: {len(ppi_sub):,}")

    if len(ppi_sub) == 0:
        logger.warning("没有PPI边！将所有SNP覆盖基因构建为孤立节点图。")
        gene_list = sorted(covered_genes)
    else:
        # 仅包含在PPI中有连接的基因
        genes_in_ppi = set(ppi_sub['gene1']) | set(ppi_sub['gene2'])
        gene_list = sorted(genes_in_ppi)

    n = len(gene_list)
    gene2idx = {g: i for i, g in enumerate(gene_list)}
    logger.info(f"最终基因节点数: {n}")

    # 构建邻接矩阵
    if len(ppi_sub) > 0:
        rows = ppi_sub['gene1'].map(gene2idx).dropna().astype(int).values
        cols = ppi_sub['gene2'].map(gene2idx).dropna().astype(int).values
        valid = (ppi_sub['gene1'].isin(gene2idx)) & (ppi_sub['gene2'].isin(gene2idx))
        ppi_sub = ppi_sub[valid]
        rows = ppi_sub['gene1'].map(gene2idx).astype(int).values
        cols = ppi_sub['gene2'].map(gene2idx).astype(int).values
        weights = ppi_sub['score'].values

        # 对称矩阵
        all_rows = np.concatenate([rows, cols])
        all_cols = np.concatenate([cols, rows])
        all_w    = np.concatenate([weights, weights])

        adj = sp.csr_matrix((all_w, (all_rows, all_cols)), shape=(n, n))
    else:
        adj = sp.eye(n, format='csr')

    # 加自环，归一化
    adj = adj + sp.eye(n, format='csr')
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.where(deg > 0, deg ** -0.5, 0.0)
    D_inv = sp.diags(deg_inv_sqrt)
    adj_norm = D_inv @ adj @ D_inv
    adj_norm = adj_norm.astype(np.float32)

    # 保存
    sp.save_npz(cache_adj, adj_norm)
    Path(cache_gene).write_text('\n'.join(gene_list))
    logger.info(f"保存PPI邻接: {cache_adj}, 稀疏度={adj_norm.nnz/(n*n)*100:.2f}%")

    return adj_norm, gene_list


# ══════════════════════════════════════════════════════════════════════════════
# Step 4: 计算基因特征矩阵
# ══════════════════════════════════════════════════════════════════════════════

def build_gene_features(trait: str, snp_to_gene: pd.DataFrame,
                         gene_list: list) -> bool:
    """
    为指定性状计算各split的基因特征矩阵。
    gene_feat[sample, gene] = mean(SNP values of SNPs mapped to that gene)
    """
    trait_graph_dir = GRAPH_DIR / trait
    trait_graph_dir.mkdir(exist_ok=True)

    # 检查缓存
    if (trait_graph_dir / 'gene_feat_train.npy').exists():
        logger.info(f"  {trait}: 基因特征已存在，跳过")
        return True

    trait_dir = PROC_DIR / trait
    meta_path = trait_dir / 'selected_snp_meta.csv'
    if not meta_path.exists():
        return False

    snp_meta = pd.read_csv(meta_path)
    gene2idx = {g: i for i, g in enumerate(gene_list)}
    n_genes  = len(gene_list)

    # 找出该性状选中的SNP中有基因映射的
    snp_meta_with_gene = snp_meta.merge(
        snp_to_gene[['snp_id', 'gene_id']], on='snp_id', how='left'
    )
    snp_meta_with_gene = snp_meta_with_gene.dropna(subset=['gene_id'])
    snp_meta_with_gene = snp_meta_with_gene[
        snp_meta_with_gene['gene_id'].isin(gene2idx)
    ]

    logger.info(f"  {trait}: {len(snp_meta_with_gene)} SNPs有基因映射 "
                f"(覆盖 {snp_meta_with_gene['gene_id'].nunique()} 基因)")

    for split in ['train', 'val', 'test']:
        X = np.load(trait_dir / f'X_{split}.npy')  # (n_samples, n_snps_pcs)
        n_samples = X.shape[0]

        gene_feat = np.zeros((n_samples, n_genes), dtype=np.float32)
        gene_count = np.zeros(n_genes, dtype=np.int32)

        for _, row in snp_meta_with_gene.iterrows():
            pcs_rank = int(row['pcs_rank'])
            gidx = gene2idx[row['gene_id']]
            if pcs_rank < X.shape[1]:
                gene_feat[:, gidx] += X[:, pcs_rank]
                gene_count[gidx] += 1

        nz = gene_count > 0
        gene_feat[:, nz] /= gene_count[nz]
        np.save(trait_graph_dir / f'gene_feat_{split}.npy', gene_feat)

    logger.info(f"  {trait}: 基因特征saved ({n_samples}, {n_genes})")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("GSTP007 基因网络构建")
    print("=" * 60)

    # 加载基础数据
    logger.info("加载基因注释...")
    gene_ann = pd.read_csv(NET_DIR / 'rice_gene_annotation.tsv', sep='\t')
    logger.info(f"基因注释: {len(gene_ann):,} 基因")

    logger.info("加载STRING PPI...")
    ppi = pd.read_csv(NET_DIR / 'rice_ppi_string700.tsv', sep='\t')
    logger.info(f"STRING PPI: {len(ppi):,} 边")

    logger.info("加载STRING别名...")
    aliases = pd.read_csv(NET_DIR / 'rice_string_aliases.tsv', sep='\t')
    logger.info(f"别名映射: {len(aliases):,} 条")

    # Step 1: 收集SNP元信息
    snp_meta_all = collect_all_snp_metadata()

    # Step 2: SNP → Gene 映射
    snp_to_gene = map_snps_to_genes(snp_meta_all, gene_ann)

    if snp_to_gene.empty:
        logger.error("SNP→Gene映射为空！检查染色体格式是否一致。")
        # Debug: 检查染色体格式
        logger.info(f"SNP meta chr: {sorted(snp_meta_all['chr'].unique()[:5])}")
        logger.info(f"Gene ann chr: {sorted(gene_ann['chr'].unique()[:5])}")
        return

    # Step 3: 构建PPI邻接矩阵
    adj, gene_list = build_ppi_adjacency(snp_to_gene, ppi, aliases)
    print(f"\n基因节点数: {len(gene_list):,}")
    print(f"PPI边数: {adj.nnz:,}, 稀疏度: {adj.nnz/(len(gene_list)**2)*100:.2f}%")

    # Step 4: 各性状的基因特征
    print("\n计算各性状基因特征...")
    for trait in TRAITS:
        build_gene_features(trait, snp_to_gene, gene_list)

    print("\n完成！图数据保存至:", GRAPH_DIR)
    print(f"基因节点: {len(gene_list):,}")

    # 验证
    for trait in TRAITS[:2]:
        feat_path = GRAPH_DIR / trait / 'gene_feat_train.npy'
        if feat_path.exists():
            feat = np.load(feat_path)
            covered = (feat.any(axis=0)).sum()
            print(f"  {trait}: gene_feat_train={feat.shape}, "
                  f"覆盖基因={covered}/{len(gene_list)}")


if __name__ == '__main__':
    main()

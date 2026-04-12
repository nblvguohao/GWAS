#!/usr/bin/env python3
"""
为GSTP007构建GCN所需的基因网络

步骤：
  1. 合并各性状的selected SNP位置，映射到水稻基因（±20kb窗口）
  2. 构建基因-基因PPI邻接矩阵（STRING v12, score>700）
  3. 对每个trait+split：计算每个基因的多维SNP特征（5维）
  4. 保存图数据（稀疏邻接 + 基因特征矩阵）

输出：
  data/processed/gstp007/graph/
    ├── ppi_adj.npz              (n_genes, n_genes) 稀疏邻接矩阵
    ├── gene_list.txt            基因ID列表
    ├── global_gene_feat_v2.npy  (n_samples, n_genes, 5) 多维基因特征缓存
    └── {trait}/
        ├── gene_feat_v2_train.npy  (n_train, n_genes, 5) 多维基因特征
        ├── gene_feat_v2_val.npy    5维: [mean, std, max, min, log_count]
        └── gene_feat_v2_test.npy

gene_feat维度说明（F=5）：
  F0 mean      — 基因区间内所有SNP的均值（平均等位基因效应）
  F1 std       — SNP值的标准差（基因内遗传多样性）
  F2 max       — SNP最大值（峰值等位基因信号）
  F3 min       — SNP最小值（最小等位基因信号）
  F4 log_count — log(SNP数+1) 归一化（基因SNP密度，结构特征）
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

# PLINK BED文件路径（用于全基因组SNP提取）
PLINK_BASE = Path('E:/GWAS/data/raw/gstp007/1495Hybrid_MSUv7')

DATA_DIR   = Path('E:/GWAS/data')
NET_DIR    = DATA_DIR / 'raw' / 'networks'
PROC_DIR   = DATA_DIR / 'processed' / 'gstp007'
GRAPH_DIR  = PROC_DIR / 'graph'
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

GENE_WINDOW = 100_000   # ±100kb SNP→gene映射窗口

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

def load_global_gene_features(snp_to_gene: pd.DataFrame, gene_list: list,
                               bim_df: pd.DataFrame) -> np.ndarray:
    """
    从BED文件读取所有30K SNP（映射到PPI基因的），构建全样本多维基因特征矩阵。

    返回: (n_samples, n_genes, 5) 多维基因特征矩阵
         5维: [mean, std, max, min, log_count]
    """
    cache_path = GRAPH_DIR / 'global_gene_feat_v2.npy'
    if cache_path.exists():
        logger.info("加载缓存多维全局基因特征矩阵...")
        feat = np.load(cache_path)
        logger.info(f"  缓存shape: {feat.shape}")
        return feat

    logger.info("从BED文件构建多维全局基因特征矩阵（覆盖所有30K SNP）...")

    try:
        from bed_reader import open_bed
    except ImportError:
        raise ImportError("请安装 bed_reader: pip install bed-reader")

    gene2idx = {g: i for i, g in enumerate(gene_list)}
    n_genes = len(gene_list)

    # 仅保留映射到PPI基因的SNP
    stg_ppi = snp_to_gene[snp_to_gene['gene_id'].isin(gene2idx)].copy()
    logger.info(f"映射到PPI基因的SNP: {len(stg_ppi):,} 行, {stg_ppi['snp_id'].nunique():,} 唯一SNP, "
                f"覆盖 {stg_ppi['gene_id'].nunique():,}/{n_genes} 基因")

    # 在BIM文件中找对应行号
    bim_snp_id = bim_df['chr'].astype(str) + '_' + bim_df['pos'].astype(str)
    snp_id_to_bim_idx = dict(zip(bim_snp_id, range(len(bim_df))))

    stg_ppi = stg_ppi.copy()
    stg_ppi['bim_idx'] = stg_ppi['snp_id'].map(snp_id_to_bim_idx)
    stg_ppi = stg_ppi.dropna(subset=['bim_idx'])
    stg_ppi['bim_idx'] = stg_ppi['bim_idx'].astype(int)
    stg_ppi = stg_ppi.drop_duplicates(subset=['snp_id'])

    logger.info(f"找到BIM索引的SNP: {len(stg_ppi):,}")

    if len(stg_ppi) == 0:
        logger.error("无法在BIM文件中找到PPI基因对应的SNP，返回零矩阵")
        bed = open_bed(str(PLINK_BASE) + '.bed', count_A1=False)
        n_samples = bed.iid_count
        return np.zeros((n_samples, n_genes, 5), dtype=np.float32)

    # 按块读取BED文件
    bed = open_bed(str(PLINK_BASE) + '.bed', count_A1=False)
    n_samples = bed.iid_count
    logger.info(f"BED文件: {n_samples} 样本, {bed.sid_count:,} SNPs")

    # 批量读取所有需要的SNP
    bim_indices = sorted(stg_ppi['bim_idx'].unique())
    logger.info(f"需要读取 {len(bim_indices)} 个SNP列...")

    # 分批读取避免内存溢出
    BATCH = 5000
    snp_data = {}  # bim_idx -> column data (n_samples,)
    for i in range(0, len(bim_indices), BATCH):
        batch_idx = bim_indices[i:i+BATCH]
        chunk = bed.read(np.s_[:, batch_idx], dtype=np.float32)
        col_mean = np.nanmean(chunk, axis=0)
        for j, bidx in enumerate(batch_idx):
            col = chunk[:, j].copy()
            col[np.isnan(col)] = col_mean[j] if not np.isnan(col_mean[j]) else 0.0
            snp_data[bidx] = col
        if (i // BATCH) % 2 == 0:
            logger.info(f"  读取进度: {min(i+BATCH, len(bim_indices))}/{len(bim_indices)} SNPs")

    # 累积统计量：sum, sum2, max, min, count
    gene_sum   = np.zeros((n_samples, n_genes), dtype=np.float64)
    gene_sum2  = np.zeros((n_samples, n_genes), dtype=np.float64)
    gene_max   = np.full((n_samples, n_genes), -np.inf, dtype=np.float32)
    gene_min   = np.full((n_samples, n_genes),  np.inf, dtype=np.float32)
    gene_count = np.zeros(n_genes, dtype=np.int32)

    for _, row in stg_ppi.iterrows():
        gidx = gene2idx.get(row['gene_id'])
        if gidx is None:
            continue
        bidx = int(row['bim_idx'])
        if bidx not in snp_data:
            continue
        col = snp_data[bidx]          # (n_samples,)
        gene_sum[:, gidx]  += col
        gene_sum2[:, gidx] += col * col
        np.maximum(gene_max[:, gidx], col, out=gene_max[:, gidx])
        np.minimum(gene_min[:, gidx], col, out=gene_min[:, gidx])
        gene_count[gidx] += 1

    # 计算5个特征
    nz = gene_count > 0                   # (n_genes,) 有SNP覆盖的基因

    # F0: mean
    feat_mean = np.zeros((n_samples, n_genes), dtype=np.float32)
    feat_mean[:, nz] = (gene_sum[:, nz] / gene_count[nz]).astype(np.float32)

    # F1: std  =  sqrt(E[x^2] - E[x]^2)，数值稳定加clip
    feat_std = np.zeros((n_samples, n_genes), dtype=np.float32)
    if nz.any():
        var = gene_sum2[:, nz] / gene_count[nz] - (gene_sum[:, nz] / gene_count[nz]) ** 2
        feat_std[:, nz] = np.sqrt(np.clip(var, 0, None)).astype(np.float32)

    # F2: max（无SNP的基因设为0）
    feat_max = np.where(nz[None, :], gene_max, 0.0).astype(np.float32)

    # F3: min（无SNP的基因设为0）
    feat_min = np.where(nz[None, :], gene_min, 0.0).astype(np.float32)

    # F4: log-normalized count（结构特征，所有样本相同）
    max_count = max(int(gene_count.max()), 1)
    feat_logcnt = (np.log1p(gene_count) / np.log1p(max_count)).astype(np.float32)
    feat_logcnt = np.broadcast_to(feat_logcnt[None, :], (n_samples, n_genes)).copy()

    # 拼接 → (n_samples, n_genes, 5)
    global_feat = np.stack(
        [feat_mean, feat_std, feat_max, feat_min, feat_logcnt], axis=-1
    )  # (n_samples, n_genes, 5)

    covered = nz.sum()
    logger.info(f"多维基因特征: {covered}/{n_genes} 基因有SNP覆盖 "
                f"({covered/n_genes*100:.1f}%), shape={global_feat.shape}")
    np.save(cache_path, global_feat)
    return global_feat


def build_gene_features(trait: str, snp_to_gene: pd.DataFrame,
                         gene_list: list,
                         global_gene_feat: np.ndarray = None,
                         bim_df: pd.DataFrame = None) -> bool:
    """
    为指定性状计算各split的多维基因特征矩阵。
    使用全部30K SNP（而非仅PCS选中的5000个），大幅提高PPI基因覆盖率。
    gene_feat[sample, gene, :] = [mean, std, max, min, log_count]
    """
    trait_graph_dir = GRAPH_DIR / trait
    trait_graph_dir.mkdir(exist_ok=True)

    # 检查v2缓存（优先）
    if (trait_graph_dir / 'gene_feat_v2_train.npy').exists():
        logger.info(f"  {trait}: 多维基因特征(v2)已存在，跳过")
        return True

    trait_dir = PROC_DIR / trait

    # 使用全局多维基因特征矩阵（基于BED文件全部30K SNP）
    if global_gene_feat is not None:
        # 按split加载样本索引，从全局特征矩阵提取子集
        for split in ['train', 'val', 'test']:
            idx_file = trait_dir / f'{split}_idx.csv'
            if not idx_file.exists():
                logger.warning(f"  {trait}: 找不到 {split}_idx.csv")
                continue
            sample_idx = pd.read_csv(idx_file)['idx'].values  # BED文件中的行号
            gene_feat = global_gene_feat[sample_idx]  # (n_split, n_genes, 5)
            np.save(trait_graph_dir / f'gene_feat_v2_{split}.npy', gene_feat)

        n_genes = global_gene_feat.shape[1]
        # 覆盖率用mean通道判断（F0）
        covered = (global_gene_feat[..., 0].any(axis=0)).sum()
        logger.info(f"  {trait}: v2基因特征saved {global_gene_feat.shape[-1]}维, "
                    f"覆盖基因={covered}/{n_genes}")
        return True

    logger.warning(f"  {trait}: 未提供global_gene_feat，跳过")
    return False


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
        logger.info(f"SNP meta chr: {sorted(snp_meta_all['chr'].unique()[:5])}")
        logger.info(f"Gene ann chr: {sorted(gene_ann['chr'].unique()[:5])}")
        return

    # Step 3: 构建PPI邻接矩阵
    adj, gene_list = build_ppi_adjacency(snp_to_gene, ppi, aliases)
    print(f"\n基因节点数: {len(gene_list):,}")
    print(f"PPI边数: {adj.nnz:,}, 稀疏度: {adj.nnz/(len(gene_list)**2)*100:.2f}%")

    # Step 4: 加载BIM文件，构建全局基因特征矩阵（使用全部30K SNP）
    logger.info("加载BIM文件...")
    bim_df = pd.read_csv(str(PLINK_BASE) + '.bim', sep='\t',
                         names=['chr', 'snp_id_raw', 'cm', 'pos', 'a1', 'a2'])
    logger.info(f"BIM: {len(bim_df):,} SNPs")

    print("\n构建全局基因特征矩阵（全部30K SNP → PPI基因）...")
    global_gene_feat = load_global_gene_features(snp_to_gene, gene_list, bim_df)

    # Step 5: 各性状的基因特征（直接从全局矩阵按split索引提取）
    print("\n按性状split提取基因特征...")
    for trait in TRAITS:
        build_gene_features(trait, snp_to_gene, gene_list,
                            global_gene_feat=global_gene_feat, bim_df=bim_df)

    print("\n完成！图数据保存至:", GRAPH_DIR)
    print(f"基因节点: {len(gene_list):,}")

    # 验证
    for trait in TRAITS:
        feat_path = GRAPH_DIR / trait / 'gene_feat_v2_train.npy'
        if feat_path.exists():
            feat = np.load(feat_path)
            covered = (feat[..., 0].any(axis=0)).sum()
            print(f"  {trait}: gene_feat_v2_train={feat.shape}, "
                  f"覆盖基因={covered}/{len(gene_list)} ({covered/len(gene_list)*100:.1f}%)")


if __name__ == '__main__':
    main()

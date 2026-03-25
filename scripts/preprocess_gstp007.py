#!/usr/bin/env python3
"""
GSTP007 真实水稻数据预处理流水线（内存高效版）

策略：
  - 流式读取BED（分块），从不将全部SNP同时加载入内存
  - Pass 1: QC统计（缺失率+MAF），记录通过QC的SNP索引
  - Pass 2: 在QC通过的SNP上计算Pearson r，选top-K
  - Pass 3: 仅读取最终选中的5000个SNP并保存

数据: GSTP007（1495样本，1,651,507 SNPs，PLINK binary）
输出: data/processed/gstp007/{trait}/X_{split}.npy 等
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging
import json
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ── 路径 ───────────────────────────────────────────────────────────────────────
DATA_DIR   = Path('E:/GWAS/data')
RAW_DIR    = DATA_DIR / 'raw' / 'gstp007'
PROC_DIR   = DATA_DIR / 'processed' / 'gstp007'
PLINK_BASE = RAW_DIR / '1495Hybrid_MSUv7'
PHENO_FILE = RAW_DIR / 'GSTP007.pheno'

# ── 目标性状 ────────────────────────────────────────────────────────────────────
TARGET_TRAITS = {
    'Plant_Height':    'BLUP_Height',
    'Grain_Length':    'BLUP_Grain_length',
    'Grain_Width':     'BLUP_Grain_width',
    'Days_to_Heading': 'BLUP_Heading_date',
    'Panicle_Length':  'BLUP_Panicle_length',
    'Grain_Weight':    'BLUP_Grain_weight',
    'Yield_per_plant': 'BLUP_Yield_per_plant',
}

# ── 超参数 ─────────────────────────────────────────────────────────────────────
MAF_THRESH     = 0.05
MISS_THRESH    = 0.10
TOP_K          = 5000
CHUNK          = 100_000   # 每批处理的SNP数
RANDOM_SEED    = 42
TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15


# ══════════════════════════════════════════════════════════════════════════════
# Pass 1 & 2: QC + PCS（分块，低内存）
# ══════════════════════════════════════════════════════════════════════════════

def qc_and_pcs_pass(bed, n_samples: int, n_snps_total: int,
                    train_idx: np.ndarray, y_train: np.ndarray,
                    maf_thresh=MAF_THRESH, miss_thresh=MISS_THRESH,
                    top_k=TOP_K, chunk=CHUNK) -> np.ndarray:
    """
    一遍扫描完成QC过滤 + Pearson|r|计算。
    返回: top_k 个选中SNP的全局索引 (升序，方便后续读取)
    """
    n_tr = len(train_idx)
    y_tr = y_train.astype(np.float64)
    y_c  = y_tr - y_tr.mean()
    y_norm = np.sqrt((y_c ** 2).sum())

    # 存储所有通过QC的 (|r|, global_snp_idx)
    abs_r_list = []
    snp_idx_list = []

    total_pass_qc = 0

    for start in range(0, n_snps_total, chunk):
        end = min(start + chunk, n_snps_total)
        # 读取当前块：shape (n_samples, chunk_size)
        chunk_geno = bed.read(
            np.s_[:, start:end],
            dtype=np.float32
        )  # NaN = missing

        # ── QC ──
        missing_rate = np.isnan(chunk_geno).mean(axis=0)
        pass_miss = missing_rate <= miss_thresh

        # 计算列均值（用于MAF估计和缺失值填充）
        col_mean = np.nanmean(chunk_geno, axis=0)
        allele_freq = col_mean / 2.0
        maf = np.minimum(allele_freq, 1 - allele_freq)
        pass_maf = maf >= maf_thresh

        pass_qc = pass_miss & pass_maf
        n_pass = pass_qc.sum()
        total_pass_qc += n_pass

        if n_pass == 0:
            if start % (chunk * 5) == 0:
                logger.info(f"  QC: {end:,}/{n_snps_total:,} — 本块0通过")
            continue

        # 取通过QC的列
        sub = chunk_geno[:, pass_qc]  # (n_samples, n_pass)
        sub_mean = col_mean[pass_qc]

        # 缺失值填充（列均值）
        for j in range(sub.shape[1]):
            nan_mask = np.isnan(sub[:, j])
            if nan_mask.any():
                sub[nan_mask, j] = sub_mean[j]

        # ── 只用训练集计算Pearson r ──
        sub_tr = sub[train_idx].astype(np.float64)  # (n_tr, n_pass)
        sub_tr_c = sub_tr - sub_tr.mean(axis=0)
        num = (sub_tr_c * y_c[:, None]).sum(axis=0)
        denom_g = np.sqrt((sub_tr_c ** 2).sum(axis=0))
        with np.errstate(invalid='ignore', divide='ignore'):
            r = np.where(denom_g > 0, num / (denom_g * y_norm), 0.0)

        abs_r = np.abs(r)

        # 全局索引（pass_qc在当前chunk中的位置 → 全局位置）
        local_pass_idx = np.where(pass_qc)[0]
        global_idx = start + local_pass_idx

        abs_r_list.append(abs_r)
        snp_idx_list.append(global_idx)

        if start % (chunk * 5) == 0:
            logger.info(f"  QC+PCS: {end:,}/{n_snps_total:,} SNPs, "
                        f"通过QC: {total_pass_qc:,}")

    if not abs_r_list:
        raise RuntimeError("没有SNP通过QC过滤！")

    all_abs_r  = np.concatenate(abs_r_list)
    all_snp_idx = np.concatenate(snp_idx_list)
    logger.info(f"  总QC通过: {total_pass_qc:,} SNPs")

    # 选top-K
    k = min(top_k, len(all_abs_r))
    top_local = np.argpartition(all_abs_r, -k)[-k:]
    top_local_sorted = top_local[np.argsort(all_abs_r[top_local])[::-1]]

    selected_global = np.sort(all_snp_idx[top_local_sorted])  # 升序，方便BED读取
    logger.info(f"  PCS选出: {len(selected_global)} SNPs, "
                f"|r| 范围 [{all_abs_r[top_local_sorted[-1]]:.4f}, "
                f"{all_abs_r[top_local_sorted[0]]:.4f}]")
    return selected_global


# ══════════════════════════════════════════════════════════════════════════════
# Pass 3: 读取选中SNP
# ══════════════════════════════════════════════════════════════════════════════

def read_selected_snps(bed, selected_global_idx: np.ndarray,
                       n_samples: int, chunk=10000) -> np.ndarray:
    """
    按全局索引批量读取选中的SNP，拼接成 (n_samples, n_selected) 矩阵。
    """
    n_sel = len(selected_global_idx)
    result = np.empty((n_samples, n_sel), dtype=np.float32)

    logger.info(f"  读取选中SNP: {n_sel}个...")
    for batch_start in range(0, n_sel, chunk):
        batch_end = min(batch_start + chunk, n_sel)
        batch_idx = selected_global_idx[batch_start:batch_end]
        # bed_reader支持按列索引读取
        sub = bed.read(np.s_[:, batch_idx], dtype=np.float32)
        # 填充缺失值
        col_mean = np.nanmean(sub, axis=0)
        for j in range(sub.shape[1]):
            nan_mask = np.isnan(sub[:, j])
            if nan_mask.any():
                sub[nan_mask, j] = col_mean[j]
        result[:, batch_start:batch_end] = sub

    logger.info(f"  读取完成: shape={result.shape}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def process_trait(bed, bim: pd.DataFrame, n_samples: int, n_snps_total: int,
                  phenotype: pd.Series, trait_name: str,
                  output_dir: Path):
    """为单个性状执行完整预处理。"""
    trait_dir = output_dir / trait_name
    trait_dir.mkdir(parents=True, exist_ok=True)

    # 过滤表型缺失
    valid_mask = phenotype.notna().values
    y_valid = phenotype[valid_mask].values.astype(np.float32)
    valid_sample_idx = np.where(valid_mask)[0]
    n_valid = len(y_valid)
    logger.info(f"  有效样本: {n_valid} (去掉 {(~valid_mask).sum()} 缺失)")

    # Train/Val/Test 划分
    all_idx = np.arange(n_valid)
    train_val_idx, test_idx = train_test_split(
        all_idx, test_size=1 - TRAIN_RATIO - VAL_RATIO,
        random_state=RANDOM_SEED
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO),
        random_state=RANDOM_SEED
    )
    logger.info(f"  划分: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # 将 valid_sample_idx 映射到全局（1495个样本中的实际行号）
    global_train_idx = valid_sample_idx[train_idx]

    # Pass 1+2: QC + PCS（基于训练集）
    logger.info(f"  Pass 1+2: QC过滤 + PCS特征选择...")
    y_train = y_valid[train_idx]
    selected_global = qc_and_pcs_pass(
        bed, n_samples, n_snps_total,
        global_train_idx, y_train
    )

    # Pass 3: 读取选中SNP
    logger.info(f"  Pass 3: 读取选中SNP...")
    # 读取所有1495个样本在选中SNP上的基因型
    geno_all = read_selected_snps(bed, selected_global, n_samples)
    # 只取有效样本
    geno_valid = geno_all[valid_sample_idx]

    # 标准化（基于训练集）
    y_mean = y_valid[train_idx].mean()
    y_std  = y_valid[train_idx].std()
    y_scaled = (y_valid - y_mean) / (y_std + 1e-8)

    geno_mean = geno_valid[train_idx].mean(axis=0)
    geno_std  = geno_valid[train_idx].std(axis=0)
    geno_std  = np.where(geno_std > 0, geno_std, 1.0)
    geno_norm = (geno_valid - geno_mean) / geno_std

    # 保存
    splits = {'train': train_idx, 'val': val_idx, 'test': test_idx}
    for split, idx in splits.items():
        np.save(trait_dir / f'X_{split}.npy',        geno_norm[idx].astype(np.float32))
        np.save(trait_dir / f'y_{split}.npy',        y_valid[idx].astype(np.float32))
        np.save(trait_dir / f'y_{split}_scaled.npy', y_scaled[idx].astype(np.float32))
        pd.DataFrame({'idx': idx}).to_csv(trait_dir / f'{split}_idx.csv', index=False)

    # 保存SNP元信息
    selected_bim = bim.iloc[selected_global].copy().reset_index(drop=True)
    selected_bim['pcs_rank'] = np.arange(len(selected_bim))
    selected_bim.to_csv(trait_dir / 'selected_snp_meta.csv', index=False)

    meta = {
        'trait': trait_name,
        'n_valid': int(n_valid),
        'n_train': int(len(train_idx)),
        'n_val':   int(len(val_idx)),
        'n_test':  int(len(test_idx)),
        'n_snps_pcs': int(len(selected_global)),
        'y_train_mean': float(y_mean),
        'y_train_std':  float(y_std),
    }
    with open(trait_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    logger.info(f"  ✓ {trait_name} 完成: X shape ({n_valid}, {len(selected_global)})")
    return meta


def main():
    from bed_reader import open_bed

    print("=" * 60)
    print("GSTP007 真实水稻数据预处理（流式高效版）")
    print("=" * 60)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    # 读取BIM/FAM（轻量元信息）
    bim = pd.read_csv(str(PLINK_BASE) + '.bim', sep='\t', header=None,
                      names=['chr', 'snp_id', 'cm', 'pos', 'a1', 'a2'])
    fam = pd.read_csv(str(PLINK_BASE) + '.fam', sep=' ', header=None,
                      names=['fam', 'id', 'pat', 'mat', 'sex', 'pheno'])
    n_snps   = len(bim)
    n_samples = len(fam)
    logger.info(f"BIM: {n_snps:,} SNPs, FAM: {n_samples} 样本")

    # 读取表型
    pheno_df = pd.read_csv(PHENO_FILE, sep=r'\s+').set_index('LINE')
    # 对齐FAM顺序
    fam_ids = fam['id'].values
    pheno_aligned = pheno_df.reindex(fam_ids)
    logger.info(f"表型对齐: {(~pheno_aligned.isnull().all(axis=1)).sum()} 个样本有表型")

    # 打开BED（lazy，不加载全部数据）
    bed = open_bed(str(PLINK_BASE) + '.bed', num_threads=4)

    all_meta = {}
    for trait_name, pheno_col in TARGET_TRAITS.items():
        if pheno_col not in pheno_aligned.columns:
            logger.warning(f"跳过 {trait_name}: {pheno_col} 不存在")
            continue
        print(f"\n{'─'*50}")
        print(f"处理性状: {trait_name} → {pheno_col}")
        print(f"{'─'*50}")
        meta = process_trait(
            bed, bim, n_samples, n_snps,
            pheno_aligned[pheno_col],
            trait_name, PROC_DIR
        )
        all_meta[trait_name] = meta

    with open(PROC_DIR / 'summary.json', 'w') as f:
        json.dump(all_meta, f, indent=2)

    print("\n" + "=" * 60)
    print("预处理完成！")
    print("=" * 60)
    print(f"{'性状':<22} {'样本':>6} {'训练':>6} {'验证':>6} {'测试':>6} {'SNP':>6}")
    print("-" * 58)
    for t, m in all_meta.items():
        print(f"{t:<22} {m['n_valid']:>6} {m['n_train']:>6} "
              f"{m['n_val']:>6} {m['n_test']:>6} {m['n_snps_pcs']:>6}")
    print(f"\n保存目录: {PROC_DIR}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""
Preprocess rice469 dataset
Includes quality control, PCS feature selection, and one-hot encoding
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import json

def quality_control(genotype_df, missing_threshold=0.1, maf_threshold=0.05):
    """
    Quality control for SNP data
    
    Args:
        genotype_df: DataFrame with samples as rows, SNPs as columns
        missing_threshold: Maximum missing rate per SNP
        maf_threshold: Minimum minor allele frequency
    
    Returns:
        Filtered genotype DataFrame
    """
    print("\n" + "="*60)
    print("Quality Control")
    print("="*60)
    
    initial_snps = genotype_df.shape[1]
    print(f"Initial SNPs: {initial_snps}")
    
    # 1. Filter by missing rate
    missing_rate = (genotype_df == -1).sum(axis=0) / len(genotype_df)
    snps_to_keep = missing_rate <= missing_threshold
    
    print(f"\nMissing rate filter (threshold={missing_threshold}):")
    print(f"  SNPs removed: {(~snps_to_keep).sum()}")
    print(f"  SNPs retained: {snps_to_keep.sum()}")
    
    genotype_df = genotype_df.loc[:, snps_to_keep]
    
    # 2. Impute missing values with mode
    print("\nImputing missing values with mode...")
    for col in genotype_df.columns:
        mode_val = genotype_df[col][genotype_df[col] != -1].mode()
        if len(mode_val) > 0:
            genotype_df.loc[genotype_df[col] == -1, col] = mode_val[0]
        else:
            genotype_df.loc[genotype_df[col] == -1, col] = 1  # Default to heterozygous
    
    # 3. Filter by MAF
    print("\nCalculating Minor Allele Frequency...")
    maf_list = []
    for col in genotype_df.columns:
        allele_counts = genotype_df[col].value_counts()
        total_alleles = len(genotype_df) * 2
        
        # Count alleles (0=AA, 1=AB, 2=BB)
        count_A = allele_counts.get(0, 0) * 2 + allele_counts.get(1, 0)
        count_B = allele_counts.get(2, 0) * 2 + allele_counts.get(1, 0)
        
        freq_A = count_A / total_alleles
        freq_B = count_B / total_alleles
        
        maf = min(freq_A, freq_B)
        maf_list.append(maf)
    
    maf_series = pd.Series(maf_list, index=genotype_df.columns)
    snps_to_keep = maf_series >= maf_threshold
    
    print(f"\nMAF filter (threshold={maf_threshold}):")
    print(f"  SNPs removed: {(~snps_to_keep).sum()}")
    print(f"  SNPs retained: {snps_to_keep.sum()}")
    
    genotype_df = genotype_df.loc[:, snps_to_keep]
    
    print(f"\nFinal SNPs after QC: {genotype_df.shape[1]} (from {initial_snps})")
    
    return genotype_df


def pcs_feature_selection(genotype_df, phenotype_df, trait_idx=0, 
                          corr_threshold=0.3, vif_threshold=10, max_snps=500):
    """
    PCS (Pearson Correlation + VIF) feature selection
    
    Args:
        genotype_df: Genotype DataFrame
        phenotype_df: Phenotype DataFrame
        trait_idx: Which trait to use for selection
        corr_threshold: Minimum absolute correlation with trait
        vif_threshold: Maximum VIF (Variance Inflation Factor)
        max_snps: Maximum number of SNPs to retain
    
    Returns:
        Selected genotype DataFrame
    """
    print("\n" + "="*60)
    print("PCS Feature Selection")
    print("="*60)
    
    trait_name = phenotype_df.columns[trait_idx]
    print(f"Target trait: {trait_name}")
    print(f"Initial SNPs: {genotype_df.shape[1]}")
    
    # Step 1: Pearson correlation filtering
    print(f"\nStep 1: Pearson correlation (threshold={corr_threshold})")
    
    y = phenotype_df.iloc[:, trait_idx].values
    correlations = []
    
    for col in genotype_df.columns:
        x = genotype_df[col].values
        corr, _ = pearsonr(x, y)
        correlations.append(abs(corr))
    
    corr_series = pd.Series(correlations, index=genotype_df.columns)
    
    # Select SNPs above threshold
    snps_above_threshold = corr_series[corr_series >= corr_threshold]
    
    print(f"  SNPs with |corr| >= {corr_threshold}: {len(snps_above_threshold)}")
    
    if len(snps_above_threshold) == 0:
        print(f"  Warning: No SNPs meet threshold, using top {max_snps}")
        selected_snps = corr_series.nlargest(max_snps).index
    elif len(snps_above_threshold) > max_snps:
        # Select top max_snps from those above threshold
        selected_snps = snps_above_threshold.nlargest(max_snps).index
    else:
        selected_snps = snps_above_threshold.index
    
    genotype_selected = genotype_df[selected_snps]
    
    print(f"  Selected SNPs: {len(selected_snps)}")
    print(f"  Correlation range: [{corr_series[selected_snps].min():.4f}, {corr_series[selected_snps].max():.4f}]")
    
    # Step 2: VIF filtering (simplified - remove highly correlated SNPs)
    print(f"\nStep 2: Removing highly correlated SNPs (simplified VIF)")
    
    # Calculate SNP-SNP correlation matrix
    snp_corr = genotype_selected.corr().abs()
    
    # Find pairs with high correlation
    upper_tri = snp_corr.where(np.triu(np.ones(snp_corr.shape), k=1).astype(bool))
    
    # Remove SNPs with correlation > 0.9 with any other SNP
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.9)]
    
    print(f"  SNPs with high inter-correlation: {len(to_drop)}")
    
    genotype_final = genotype_selected.drop(columns=to_drop)
    
    print(f"\nFinal selected SNPs: {genotype_final.shape[1]}")
    
    return genotype_final


def one_hot_encode(genotype_df):
    """
    One-hot encode genotype data
    
    Args:
        genotype_df: Genotype DataFrame with values 0, 1, 2
    
    Returns:
        One-hot encoded array (n_samples, n_snps, 3)
    """
    print("\n" + "="*60)
    print("One-Hot Encoding")
    print("="*60)
    
    n_samples, n_snps = genotype_df.shape
    print(f"Input shape: {genotype_df.shape}")
    
    # Create one-hot encoded array
    one_hot = np.zeros((n_samples, n_snps, 3), dtype=np.float32)
    
    genotype_array = genotype_df.values
    
    for i in range(n_samples):
        for j in range(n_snps):
            genotype = int(genotype_array[i, j])
            if 0 <= genotype <= 2:
                one_hot[i, j, genotype] = 1.0
    
    print(f"Output shape: {one_hot.shape}")
    print(f"Memory size: {one_hot.nbytes / 1e6:.2f} MB")
    
    return one_hot


def main():
    print("="*60)
    print("Rice469 Preprocessing Pipeline")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    data_dir = Path("data/raw/cropgs/rice469")
    
    genotype_df = pd.read_csv(data_dir / "rice469_genotype.csv", index_col=0)
    phenotype_df = pd.read_csv(data_dir / "rice469_phenotype.csv", index_col=0)
    
    print(f"Genotype shape: {genotype_df.shape}")
    print(f"Phenotype shape: {phenotype_df.shape}")
    
    # Quality control
    genotype_qc = quality_control(genotype_df, missing_threshold=0.1, maf_threshold=0.05)
    
    # PCS feature selection
    genotype_selected = pcs_feature_selection(
        genotype_qc, phenotype_df, 
        trait_idx=0,  # Use first trait (GrainYield)
        corr_threshold=0.1,  # Lowered for sample data
        max_snps=500
    )
    
    # One-hot encoding
    genotype_onehot = one_hot_encode(genotype_selected)
    
    # Standardize phenotypes
    print("\n" + "="*60)
    print("Standardizing Phenotypes")
    print("="*60)
    
    scaler = StandardScaler()
    phenotype_scaled = scaler.fit_transform(phenotype_df.values)
    
    print(f"Phenotype shape: {phenotype_scaled.shape}")
    print(f"Mean: {phenotype_scaled.mean(axis=0)}")
    print(f"Std: {phenotype_scaled.std(axis=0)}")
    
    # Save processed data
    output_dir = Path("data/processed/rice469")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Saving Processed Data")
    print("="*60)
    
    # Save as numpy arrays
    np.save(output_dir / "genotype_onehot.npy", genotype_onehot)
    np.save(output_dir / "phenotype_scaled.npy", phenotype_scaled)
    
    # Save SNP and sample IDs
    np.save(output_dir / "snp_ids.npy", genotype_selected.columns.values)
    np.save(output_dir / "sample_ids.npy", genotype_selected.index.values)
    
    # Save metadata
    metadata = {
        'n_samples': genotype_onehot.shape[0],
        'n_snps': genotype_onehot.shape[1],
        'n_traits': phenotype_scaled.shape[1],
        'trait_names': list(phenotype_df.columns),
        'sample_ids': list(genotype_selected.index),
        'snp_ids': list(genotype_selected.columns),
        'preprocessing': {
            'missing_threshold': 0.1,
            'maf_threshold': 0.05,
            'pcs_corr_threshold': 0.1,
            'max_snps': 500
        }
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved: {output_dir / 'genotype_onehot.npy'}")
    print(f"✓ Saved: {output_dir / 'phenotype_scaled.npy'}")
    print(f"✓ Saved: {output_dir / 'snp_ids.npy'}")
    print(f"✓ Saved: {output_dir / 'sample_ids.npy'}")
    print(f"✓ Saved: {output_dir / 'metadata.json'}")
    
    # Summary
    print("\n" + "="*60)
    print("Preprocessing Summary")
    print("="*60)
    print(f"Samples: {metadata['n_samples']}")
    print(f"SNPs: {metadata['n_snps']}")
    print(f"Traits: {metadata['n_traits']}")
    print(f"Trait names: {', '.join(metadata['trait_names'])}")
    print(f"\nData saved to: {output_dir}")
    
    print("\n" + "="*60)
    print("Next Steps")
    print("="*60)
    print("1. Build networks: python scripts/build_networks.py")
    print("2. Create data splits: python scripts/create_splits.py")
    print("3. Train model: python scripts/train_rice469.py")
    
    return 0


if __name__ == '__main__':
    exit(main())

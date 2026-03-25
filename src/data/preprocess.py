"""
SNP preprocessing pipeline for PlantHGNN
Implements PCS feature selection and quality control
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SNPPreprocessor:
    """
    SNP preprocessing pipeline
    - Quality control (missing rate, MAF filtering)
    - PCS feature selection (Pearson correlation + VIF)
    - One-hot encoding
    """
    
    def __init__(self, missing_threshold=0.1, maf_threshold=0.05):
        self.missing_threshold = missing_threshold
        self.maf_threshold = maf_threshold
        self.selected_snps = {}
    
    def load_genotype(self, genotype_path):
        """
        Load genotype data
        Expected format: CSV with samples as rows, SNPs as columns
        Values: 0 (AA), 1 (AB), 2 (BB), -1 or NaN (missing)
        """
        logger.info(f"Loading genotype from {genotype_path}")
        geno = pd.read_csv(genotype_path, index_col=0)
        logger.info(f"Loaded genotype: {geno.shape[0]} samples × {geno.shape[1]} SNPs")
        return geno
    
    def load_phenotype(self, phenotype_path):
        """
        Load phenotype data
        Expected format: CSV with samples as rows, traits as columns
        """
        logger.info(f"Loading phenotype from {phenotype_path}")
        pheno = pd.read_csv(phenotype_path, index_col=0)
        logger.info(f"Loaded phenotype: {pheno.shape[0]} samples × {pheno.shape[1]} traits")
        return pheno
    
    def quality_control(self, genotype):
        """
        Step 1: Quality control
        - Filter SNPs with missing rate > threshold
        - Filter SNPs with MAF < threshold
        - Filter samples with missing rate > threshold
        """
        logger.info("Performing quality control...")
        n_snps_before = genotype.shape[1]
        n_samples_before = genotype.shape[0]
        
        # Replace -1 with NaN
        genotype = genotype.replace(-1, np.nan)
        
        # Filter SNPs by missing rate
        snp_missing_rate = genotype.isnull().sum(axis=0) / len(genotype)
        valid_snps = snp_missing_rate <= self.missing_threshold
        genotype = genotype.loc[:, valid_snps]
        logger.info(f"Removed {(~valid_snps).sum()} SNPs with missing rate > {self.missing_threshold}")
        
        # Filter SNPs by MAF
        maf = genotype.apply(lambda x: min(x.mean() / 2, 1 - x.mean() / 2), axis=0)
        valid_maf = maf >= self.maf_threshold
        genotype = genotype.loc[:, valid_maf]
        logger.info(f"Removed {(~valid_maf).sum()} SNPs with MAF < {self.maf_threshold}")
        
        # Filter samples by missing rate
        sample_missing_rate = genotype.isnull().sum(axis=1) / genotype.shape[1]
        valid_samples = sample_missing_rate <= self.missing_threshold
        genotype = genotype.loc[valid_samples, :]
        logger.info(f"Removed {(~valid_samples).sum()} samples with missing rate > {self.missing_threshold}")
        
        logger.info(f"QC complete: {genotype.shape[0]} samples × {genotype.shape[1]} SNPs")
        logger.info(f"Filtered: {n_samples_before - genotype.shape[0]} samples, {n_snps_before - genotype.shape[1]} SNPs")
        
        return genotype
    
    def pcs_feature_selection(self, genotype, phenotype, trait_name, 
                              corr_threshold=0.3, vif_threshold=10):
        """
        Step 2: PCS (Pearson Correlation + VIF) feature selection
        Select SNPs correlated with phenotype and remove collinear SNPs
        """
        logger.info(f"PCS feature selection for trait: {trait_name}")
        
        if trait_name not in phenotype.columns:
            raise ValueError(f"Trait {trait_name} not found in phenotype data")
        
        # Align samples
        common_samples = genotype.index.intersection(phenotype.index)
        geno = genotype.loc[common_samples]
        pheno = phenotype.loc[common_samples, trait_name]
        
        # Remove samples with missing phenotype
        valid_pheno = ~pheno.isnull()
        geno = geno.loc[valid_pheno]
        pheno = pheno.loc[valid_pheno]
        
        # Impute missing genotypes with mean
        geno = geno.fillna(geno.mean())
        
        # Step 2.1: Pearson correlation filtering
        logger.info(f"Computing Pearson correlations (threshold: {corr_threshold})...")
        correlations = []
        for snp in tqdm(geno.columns, desc="Computing correlations"):
            corr, pval = pearsonr(geno[snp], pheno)
            correlations.append({'snp': snp, 'corr': corr, 'abs_corr': abs(corr), 'pval': pval})
        
        corr_df = pd.DataFrame(correlations)
        selected_by_corr = corr_df[corr_df['abs_corr'] >= corr_threshold]['snp'].tolist()
        logger.info(f"Selected {len(selected_by_corr)} SNPs by correlation")
        
        if len(selected_by_corr) == 0:
            logger.warning("No SNPs passed correlation threshold!")
            return []
        
        # Step 2.2: VIF-based collinearity removal
        logger.info(f"Removing collinear SNPs (VIF threshold: {vif_threshold})...")
        geno_selected = geno[selected_by_corr]
        
        # Iteratively remove high VIF SNPs
        selected_snps = selected_by_corr.copy()
        max_iterations = 100
        iteration = 0
        
        while iteration < max_iterations:
            if len(selected_snps) <= 1:
                break
            
            X = geno_selected[selected_snps].values
            vif_values = []
            
            for i in range(X.shape[1]):
                try:
                    vif = variance_inflation_factor(X, i)
                    vif_values.append(vif)
                except:
                    vif_values.append(np.inf)
            
            max_vif = max(vif_values)
            if max_vif <= vif_threshold:
                break
            
            # Remove SNP with highest VIF
            max_vif_idx = vif_values.index(max_vif)
            removed_snp = selected_snps.pop(max_vif_idx)
            iteration += 1
        
        logger.info(f"Final selection: {len(selected_snps)} SNPs after VIF filtering")
        
        return selected_snps
    
    def one_hot_encode(self, genotype):
        """
        Step 3: One-hot encoding
        AA (0) → [1, 0, 0]
        AB (1) → [0, 1, 0]
        BB (2) → [0, 0, 1]
        Missing → [0, 0, 0]
        """
        logger.info("One-hot encoding SNPs...")
        
        n_samples, n_snps = genotype.shape
        encoded = np.zeros((n_samples, n_snps, 3))
        
        for i, snp in enumerate(genotype.columns):
            values = genotype[snp].values
            for j, val in enumerate(values):
                if pd.isna(val):
                    encoded[j, i, :] = [0, 0, 0]
                elif val == 0:
                    encoded[j, i, :] = [1, 0, 0]
                elif val == 1:
                    encoded[j, i, :] = [0, 1, 0]
                elif val == 2:
                    encoded[j, i, :] = [0, 0, 1]
        
        return encoded
    
    def preprocess_dataset(self, genotype_path, phenotype_path, output_dir, 
                          dataset_name, traits=None):
        """
        Complete preprocessing pipeline for a dataset
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        genotype = self.load_genotype(genotype_path)
        phenotype = self.load_phenotype(phenotype_path)
        
        # Quality control
        genotype = self.quality_control(genotype)
        
        # Align samples
        common_samples = genotype.index.intersection(phenotype.index)
        genotype = genotype.loc[common_samples]
        phenotype = phenotype.loc[common_samples]
        
        logger.info(f"Aligned data: {len(common_samples)} common samples")
        
        # PCS feature selection for each trait
        if traits is None:
            traits = phenotype.columns.tolist()
        
        for trait in traits:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing trait: {trait}")
            logger.info(f"{'='*60}")
            
            selected_snps = self.pcs_feature_selection(genotype, phenotype, trait)
            
            if len(selected_snps) == 0:
                logger.warning(f"No SNPs selected for trait {trait}, skipping...")
                continue
            
            self.selected_snps[trait] = selected_snps
            
            # Extract selected SNPs
            geno_selected = genotype[selected_snps]
            
            # One-hot encoding
            encoded = self.one_hot_encode(geno_selected)
            
            # Save processed data
            trait_dir = output_dir / dataset_name / trait
            trait_dir.mkdir(parents=True, exist_ok=True)
            
            np.save(trait_dir / 'snp_matrix.npy', encoded)
            phenotype[[trait]].to_csv(trait_dir / 'phenotype.csv')
            
            # Save SNP metadata
            snp_metadata = pd.DataFrame({
                'snp_id': selected_snps,
                'index': range(len(selected_snps))
            })
            snp_metadata.to_csv(trait_dir / 'snp_metadata.csv', index=False)
            
            logger.info(f"Saved processed data to {trait_dir}")
            logger.info(f"  - SNP matrix: {encoded.shape}")
            logger.info(f"  - Phenotype: {phenotype[[trait]].shape}")
        
        logger.info(f"\nPreprocessing complete for {dataset_name}!")
        return self.selected_snps


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess SNP data for PlantHGNN')
    parser.add_argument('--genotype', required=True, help='Path to genotype CSV')
    parser.add_argument('--phenotype', required=True, help='Path to phenotype CSV')
    parser.add_argument('--output-dir', default='data/processed', help='Output directory')
    parser.add_argument('--dataset-name', required=True, help='Dataset name (e.g., rice469)')
    parser.add_argument('--traits', nargs='+', help='Traits to process (default: all)')
    parser.add_argument('--missing-threshold', type=float, default=0.1)
    parser.add_argument('--maf-threshold', type=float, default=0.05)
    parser.add_argument('--corr-threshold', type=float, default=0.3)
    parser.add_argument('--vif-threshold', type=float, default=10)
    
    args = parser.parse_args()
    
    preprocessor = SNPPreprocessor(
        missing_threshold=args.missing_threshold,
        maf_threshold=args.maf_threshold
    )
    
    preprocessor.preprocess_dataset(
        genotype_path=args.genotype,
        phenotype_path=args.phenotype,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        traits=args.traits
    )


if __name__ == '__main__':
    main()

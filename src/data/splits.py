"""
Data splitting strategies for PlantHGNN
Implements random, chromosome, and line-based splits
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, StratifiedKFold
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Data splitting strategies for genomic prediction
    - Random split: Standard k-fold cross-validation
    - Chromosome split: Leave-one-chromosome-out
    - Line split: Based on population structure (avoid relatedness leakage)
    """
    
    def __init__(self, n_folds=5, random_seed=42):
        self.n_folds = n_folds
        self.random_seed = random_seed
    
    def random_split(self, n_samples, stratify_labels=None):
        """
        Random k-fold split
        
        Args:
            n_samples: Number of samples
            stratify_labels: Optional labels for stratified split
        
        Returns:
            List of (train_idx, val_idx, test_idx) tuples
        """
        logger.info(f"Performing random {self.n_folds}-fold split...")
        
        indices = np.arange(n_samples)
        
        if stratify_labels is not None:
            kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, 
                                random_state=self.random_seed)
            splits = list(kf.split(indices, stratify_labels))
        else:
            kf = KFold(n_splits=self.n_folds, shuffle=True, 
                      random_state=self.random_seed)
            splits = list(kf.split(indices))
        
        # Convert to train/val/test splits
        result = []
        for fold_idx, (train_val_idx, test_idx) in enumerate(splits):
            # Further split train_val into train and val (80/20)
            n_val = len(train_val_idx) // 5
            np.random.seed(self.random_seed + fold_idx)
            np.random.shuffle(train_val_idx)
            
            val_idx = train_val_idx[:n_val]
            train_idx = train_val_idx[n_val:]
            
            result.append({
                'fold': fold_idx,
                'train': train_idx.tolist(),
                'val': val_idx.tolist(),
                'test': test_idx.tolist()
            })
            
            logger.info(f"Fold {fold_idx}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        
        return result
    
    def chromosome_split(self, snp_metadata, n_folds=None):
        """
        Leave-one-chromosome-out split
        
        Args:
            snp_metadata: DataFrame with 'chromosome' column
            n_folds: Number of folds (default: number of chromosomes)
        
        Returns:
            List of (train_idx, val_idx, test_idx) tuples
        """
        logger.info("Performing chromosome-based split...")
        
        if 'chromosome' not in snp_metadata.columns:
            raise ValueError("SNP metadata must contain 'chromosome' column")
        
        chromosomes = sorted(snp_metadata['chromosome'].unique())
        
        if n_folds is None:
            n_folds = len(chromosomes)
        
        logger.info(f"Found {len(chromosomes)} chromosomes")
        
        result = []
        for fold_idx, test_chr in enumerate(chromosomes[:n_folds]):
            # Test set: samples with SNPs on test chromosome
            test_idx = snp_metadata[snp_metadata['chromosome'] == test_chr].index.tolist()
            
            # Train+val set: remaining samples
            train_val_idx = snp_metadata[snp_metadata['chromosome'] != test_chr].index.tolist()
            
            # Split train_val
            n_val = len(train_val_idx) // 5
            np.random.seed(self.random_seed + fold_idx)
            np.random.shuffle(train_val_idx)
            
            val_idx = train_val_idx[:n_val]
            train_idx = train_val_idx[n_val:]
            
            result.append({
                'fold': fold_idx,
                'test_chromosome': test_chr,
                'train': train_idx,
                'val': val_idx,
                'test': test_idx
            })
            
            logger.info(f"Fold {fold_idx} (chr {test_chr}): train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        
        return result
    
    def line_split(self, population_structure, n_folds=None):
        """
        Line-based split using population structure
        Ensures train and test sets have no direct relatedness
        
        Args:
            population_structure: DataFrame with 'sample_id' and 'population' columns
            n_folds: Number of folds (default: number of populations)
        
        Returns:
            List of (train_idx, val_idx, test_idx) tuples
        """
        logger.info("Performing line-based split...")
        
        if 'population' not in population_structure.columns:
            raise ValueError("Population structure must contain 'population' column")
        
        populations = sorted(population_structure['population'].unique())
        
        if n_folds is None:
            n_folds = min(len(populations), self.n_folds)
        
        logger.info(f"Found {len(populations)} populations, using {n_folds} folds")
        
        # Assign populations to folds
        np.random.seed(self.random_seed)
        pop_to_fold = {}
        for i, pop in enumerate(populations):
            pop_to_fold[pop] = i % n_folds
        
        result = []
        for fold_idx in range(n_folds):
            # Test set: samples from populations assigned to this fold
            test_pops = [p for p, f in pop_to_fold.items() if f == fold_idx]
            test_idx = population_structure[
                population_structure['population'].isin(test_pops)
            ].index.tolist()
            
            # Train+val set: remaining samples
            train_val_idx = population_structure[
                ~population_structure['population'].isin(test_pops)
            ].index.tolist()
            
            # Split train_val
            n_val = len(train_val_idx) // 5
            np.random.seed(self.random_seed + fold_idx)
            np.random.shuffle(train_val_idx)
            
            val_idx = train_val_idx[:n_val]
            train_idx = train_val_idx[n_val:]
            
            result.append({
                'fold': fold_idx,
                'test_populations': test_pops,
                'train': train_idx,
                'val': val_idx,
                'test': test_idx
            })
            
            logger.info(f"Fold {fold_idx}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        
        return result
    
    def save_splits(self, splits, output_path):
        """Save splits to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        splits_serializable = convert_types(splits)
        
        with open(output_path, 'w') as f:
            json.dump(splits_serializable, f, indent=2)
        
        logger.info(f"Saved splits to {output_path}")
    
    def load_splits(self, split_path):
        """Load splits from JSON file"""
        with open(split_path, 'r') as f:
            splits = json.load(f)
        
        logger.info(f"Loaded {len(splits)} folds from {split_path}")
        return splits


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Create data splits for PlantHGNN')
    parser.add_argument('--strategy', required=True,
                       choices=['random', 'chromosome', 'line'],
                       help='Split strategy')
    parser.add_argument('--n-samples', type=int, help='Number of samples (for random split)')
    parser.add_argument('--snp-metadata', help='SNP metadata CSV (for chromosome split)')
    parser.add_argument('--population-structure', help='Population structure CSV (for line split)')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    splitter = DataSplitter(n_folds=args.n_folds, random_seed=args.seed)
    
    if args.strategy == 'random':
        if args.n_samples is None:
            raise ValueError("--n-samples required for random split")
        splits = splitter.random_split(args.n_samples)
    
    elif args.strategy == 'chromosome':
        if args.snp_metadata is None:
            raise ValueError("--snp-metadata required for chromosome split")
        snp_metadata = pd.read_csv(args.snp_metadata)
        splits = splitter.chromosome_split(snp_metadata)
    
    elif args.strategy == 'line':
        if args.population_structure is None:
            raise ValueError("--population-structure required for line split")
        pop_structure = pd.read_csv(args.population_structure)
        splits = splitter.line_split(pop_structure)
    
    splitter.save_splits(splits, args.output)
    logger.info("Split creation complete!")


if __name__ == '__main__':
    main()

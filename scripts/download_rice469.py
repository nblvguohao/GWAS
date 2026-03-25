#!/usr/bin/env python
"""
Download and prepare rice469 dataset
Handles CropGS-Hub data, STRING networks, and GO annotations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import gzip
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

def download_file(url, output_path, description="Downloading"):
    """Download file with progress bar"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=description,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)
        
        print(f"✓ Downloaded: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to download {url}: {e}")
        return False


def download_string_network():
    """Download STRING PPI network for rice (Oryza sativa, taxid: 4530)"""
    print("\n" + "="*60)
    print("Downloading STRING PPI Network for Rice")
    print("="*60)
    
    # STRING v12.0 for Oryza sativa
    url = "https://stringdb-downloads.org/download/protein.links.v12.0/4530.protein.links.v12.0.txt.gz"
    output_file = "data/raw/networks/4530.protein.links.v12.0.txt.gz"
    
    success = download_file(url, output_file, "STRING PPI Network")
    
    if success:
        # Extract and preview
        print("\nExtracting and previewing...")
        with gzip.open(output_file, 'rt') as f:
            lines = [next(f) for _ in range(5)]
            print("First 5 lines:")
            for line in lines:
                print(f"  {line.strip()}")
    
    return success


def download_go_annotations():
    """Download GO annotations for rice"""
    print("\n" + "="*60)
    print("Downloading GO Annotations for Rice")
    print("="*60)
    
    # Rice GO annotations from Gene Ontology Consortium
    url = "http://current.geneontology.org/annotations/osa.gaf.gz"
    output_file = "data/raw/annotations/osa.gaf.gz"
    
    success = download_file(url, output_file, "GO Annotations")
    
    if success:
        print("\nExtracting and previewing...")
        with gzip.open(output_file, 'rt') as f:
            # Skip comment lines
            lines = []
            for line in f:
                if not line.startswith('!'):
                    lines.append(line)
                    if len(lines) >= 3:
                        break
            
            print("First 3 annotation lines:")
            for line in lines:
                print(f"  {line.strip()[:100]}...")
    
    return success


def create_sample_rice469_data():
    """
    Create sample rice469 data structure
    Note: Real data should be downloaded from CropGS-Hub manually
    """
    print("\n" + "="*60)
    print("Creating Sample Rice469 Data Structure")
    print("="*60)
    print("\n⚠️  NOTE: This creates SAMPLE data for testing.")
    print("Real rice469 data should be downloaded from:")
    print("https://iagr.genomics.cn/CropGS")
    print("\nExpected files:")
    print("  - rice469_genotype.csv  (469 samples × 5291 SNPs)")
    print("  - rice469_phenotype.csv (469 samples × 6 traits)")
    
    output_dir = Path("data/raw/cropgs/rice469")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample genotype data (smaller for testing)
    n_samples = 469
    n_snps = 1000  # Reduced from 5291 for testing
    
    print(f"\nCreating sample genotype data: {n_samples} × {n_snps}")
    
    # Sample IDs
    sample_ids = [f"Sample_{i:03d}" for i in range(n_samples)]
    snp_ids = [f"SNP_{i:04d}" for i in range(n_snps)]
    
    # Generate genotype data (0, 1, 2, or -1 for missing)
    genotype = np.random.choice([0, 1, 2, -1], size=(n_samples, n_snps), p=[0.3, 0.4, 0.25, 0.05])
    
    genotype_df = pd.DataFrame(genotype, index=sample_ids, columns=snp_ids)
    genotype_df.to_csv(output_dir / "rice469_genotype.csv")
    print(f"✓ Created: {output_dir / 'rice469_genotype.csv'}")
    
    # Create sample phenotype data
    trait_names = ['GrainYield', 'PlantHeight', 'TillerNumber', 'GrainLength', 'GrainWidth', 'ThousandGrainWeight']
    
    print(f"\nCreating sample phenotype data: {n_samples} × {len(trait_names)}")
    
    # Generate phenotype data with some correlation to genotypes
    phenotype = np.random.randn(n_samples, len(trait_names))
    
    # Add some genetic signal
    for i in range(len(trait_names)):
        genetic_effect = genotype[:, i*10:(i+1)*10].mean(axis=1)
        phenotype[:, i] += genetic_effect * 0.5
    
    phenotype_df = pd.DataFrame(phenotype, index=sample_ids, columns=trait_names)
    phenotype_df.to_csv(output_dir / "rice469_phenotype.csv")
    print(f"✓ Created: {output_dir / 'rice469_phenotype.csv'}")
    
    # Create metadata
    metadata = {
        'dataset': 'rice469',
        'species': 'Oryza sativa',
        'n_samples': n_samples,
        'n_snps': n_snps,
        'n_traits': len(trait_names),
        'trait_names': trait_names,
        'note': 'This is SAMPLE data for testing. Replace with real CropGS-Hub data.',
        'real_data_url': 'https://iagr.genomics.cn/CropGS',
        'real_n_snps': 5291
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Created: {output_dir / 'metadata.json'}")
    
    print("\n" + "="*60)
    print("Sample Data Summary")
    print("="*60)
    print(f"Genotype shape: {genotype_df.shape}")
    print(f"Phenotype shape: {phenotype_df.shape}")
    print(f"\nGenotype preview:")
    print(genotype_df.iloc[:5, :5])
    print(f"\nPhenotype preview:")
    print(phenotype_df.head())
    print(f"\nMissing rate: {(genotype == -1).sum() / genotype.size * 100:.2f}%")
    
    return True


def main():
    print("="*60)
    print("Rice469 Data Download Script")
    print("="*60)
    
    # Download STRING network
    string_success = download_string_network()
    
    # Download GO annotations
    go_success = download_go_annotations()
    
    # Create sample rice469 data
    # Note: Real data should be downloaded manually from CropGS-Hub
    rice_success = create_sample_rice469_data()
    
    # Summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    print(f"STRING Network: {'✓ Success' if string_success else '✗ Failed'}")
    print(f"GO Annotations: {'✓ Success' if go_success else '✗ Failed'}")
    print(f"Rice469 Data: {'✓ Created (sample)' if rice_success else '✗ Failed'}")
    
    print("\n" + "="*60)
    print("Next Steps")
    print("="*60)
    print("1. [Optional] Replace sample data with real CropGS-Hub data")
    print("2. Run preprocessing: python scripts/preprocess_rice469.py")
    print("3. Build networks: python scripts/build_networks.py")
    print("4. Train model: python scripts/train_rice469.py")
    
    if string_success and go_success and rice_success:
        print("\n✓ All downloads completed successfully!")
        return 0
    else:
        print("\n⚠️  Some downloads failed. Check errors above.")
        return 1


if __name__ == '__main__':
    exit(main())

#!/usr/bin/env python
"""
Create train/val/test splits for rice469
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json

def create_random_split(n_samples, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Create random train/val/test split"""
    np.random.seed(seed)
    
    indices = np.random.permutation(n_samples)
    
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size:train_size+val_size].tolist()
    test_indices = indices[train_size+val_size:].tolist()
    
    return {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }


def main():
    print("="*60)
    print("Data Split Creation Script")
    print("="*60)
    
    # Load metadata
    metadata_file = Path("data/processed/rice469/metadata.json")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    n_samples = metadata['n_samples']
    
    print(f"\nTotal samples: {n_samples}")
    
    # Create split
    print("\nCreating random split (70/15/15)...")
    split = create_random_split(n_samples, train_ratio=0.7, val_ratio=0.15, seed=42)
    
    print(f"  Train: {len(split['train'])} samples")
    print(f"  Val: {len(split['val'])} samples")
    print(f"  Test: {len(split['test'])} samples")
    
    # Save split
    output_file = Path("data/processed/rice469/split.json")
    
    with open(output_file, 'w') as f:
        json.dump(split, f, indent=2)
    
    print(f"\n✓ Split saved to: {output_file}")
    
    print("\n" + "="*60)
    print("Next Steps")
    print("="*60)
    print("1. Train model: python scripts/train_rice469.py")
    
    return 0


if __name__ == '__main__':
    exit(main())

"""
PyTorch Geometric Dataset for PlantHGNN
Integrates SNP features, biological networks, and phenotypes
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.data import Data, Dataset
import json
import pickle


class PlantGPDataset(Dataset):
    """
    Dataset for plant genomic prediction with graph neural networks
    
    Combines:
    - SNP genotype data (one-hot encoded)
    - Biological networks (PPI, GO, KEGG)
    - Gene functional/structural features
    - Phenotype values
    """
    
    def __init__(self, 
                 root,
                 dataset_name,
                 split='train',
                 split_file=None,
                 fold=0,
                 transform=None,
                 pre_transform=None):
        """
        Args:
            root: Root directory containing processed data
            dataset_name: Name of dataset (e.g., 'rice469')
            split: 'train', 'val', or 'test'
            split_file: Path to split indices JSON file
            fold: Which fold to use (for cross-validation)
            transform: Optional transform to apply to data
            pre_transform: Optional pre-transform
        """
        self.dataset_name = dataset_name
        self.split = split
        self.split_file = split_file
        self.fold = fold
        
        super().__init__(root, transform, pre_transform)
        
        # Load split indices
        if split_file is not None:
            with open(split_file, 'r') as f:
                split_data = json.load(f)
            self.indices = split_data[f'fold_{fold}'][split]
        else:
            # Use all samples if no split file provided
            self.indices = list(range(len(self.snp_data)))
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return [
            f'{self.dataset_name}_snp.pt',
            f'{self.dataset_name}_phenotype.pt',
            f'{self.dataset_name}_graphs.pkl',
            f'{self.dataset_name}_metadata.json'
        ]
    
    def download(self):
        pass
    
    def process(self):
        pass
    
    def len(self):
        return len(self.indices)
    
    def get(self, idx):
        """
        Get a single sample
        
        Returns:
            Data object containing:
            - snp_data: One-hot encoded SNP features [n_snps, 3]
            - phenotype: Trait values [n_traits]
            - graph_data: Dictionary with network information
        """
        actual_idx = self.indices[idx]
        
        # Load SNP data
        snp_file = self.processed_paths[0]
        snp_data = torch.load(snp_file)
        sample_snp = snp_data[actual_idx]  # [n_snps, 3]
        
        # Load phenotype
        pheno_file = self.processed_paths[1]
        phenotype = torch.load(pheno_file)
        sample_pheno = phenotype[actual_idx]  # [n_traits]
        
        # Load graph data (shared across all samples)
        graph_file = self.processed_paths[2]
        with open(graph_file, 'rb') as f:
            graph_data = pickle.load(f)
        
        # Create Data object
        data = Data(
            snp_data=sample_snp,
            phenotype=sample_pheno,
            idx=actual_idx
        )
        
        # Add graph information
        data.node_features = graph_data['node_features']
        data.edge_index_list = graph_data['edge_index_list']
        data.random_walk_features = graph_data.get('random_walk_features')
        data.pagerank_scores = graph_data.get('pagerank_scores')
        data.gene_set_matrix = graph_data.get('gene_set_matrix')
        
        return data


class PlantGPDataLoader:
    """
    Custom DataLoader for PlantHGNN
    Handles batching of SNP data and shared graph structures
    """
    
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        self.indices = list(range(len(dataset)))
        
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i+self.batch_size]
            batch_data = [self.dataset[idx] for idx in batch_indices]
            
            # Stack SNP data and phenotypes
            snp_batch = torch.stack([d.snp_data for d in batch_data])
            pheno_batch = torch.stack([d.phenotype for d in batch_data])
            
            # Graph data is shared, just take from first sample
            graph_data = {
                'node_features': batch_data[0].node_features,
                'edge_index_list': batch_data[0].edge_index_list,
                'random_walk_features': batch_data[0].random_walk_features,
                'pagerank_scores': batch_data[0].pagerank_scores,
                'gene_set_matrix': batch_data[0].gene_set_matrix
            }
            
            yield snp_batch, graph_data, pheno_batch
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def prepare_dataset(genotype_file, phenotype_file, network_dir, output_dir, dataset_name):
    """
    Prepare dataset for PlantGPDataset
    
    Args:
        genotype_file: Path to preprocessed genotype file (one-hot encoded)
        phenotype_file: Path to phenotype CSV
        network_dir: Directory containing network files
        output_dir: Output directory for processed files
        dataset_name: Name of dataset
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Preparing {dataset_name} dataset...")
    
    # Load genotype data (already one-hot encoded from preprocessing)
    print("Loading genotype data...")
    genotype = np.load(genotype_file)  # [n_samples, n_snps, 3]
    snp_tensor = torch.from_numpy(genotype).float()
    
    # Load phenotype data
    print("Loading phenotype data...")
    phenotype_df = pd.read_csv(phenotype_file, index_col=0)
    phenotype_tensor = torch.from_numpy(phenotype_df.values).float()
    
    # Load network data
    print("Loading network data...")
    network_files = {
        'ppi': network_dir / f'{dataset_name}_ppi.pt',
        'go': network_dir / f'{dataset_name}_go.pt',
        'kegg': network_dir / f'{dataset_name}_kegg.pt'
    }
    
    edge_index_list = []
    for net_type, net_file in network_files.items():
        if net_file.exists():
            net_data = torch.load(net_file)
            edge_index_list.append(net_data.edge_index)
            print(f"  Loaded {net_type}: {net_data.num_nodes} nodes, {net_data.num_edges} edges")
    
    # Create node features (use first network's node features)
    if len(edge_index_list) > 0:
        net_data = torch.load(network_files['ppi'])
        node_features = net_data.x if hasattr(net_data, 'x') else torch.eye(net_data.num_nodes)
    else:
        # Fallback: create dummy node features
        n_genes = genotype.shape[1]  # Assume n_snps = n_genes for now
        node_features = torch.eye(n_genes)
    
    # Create graph data dictionary
    graph_data = {
        'node_features': node_features,
        'edge_index_list': edge_index_list,
        'random_walk_features': None,  # Will be computed if needed
        'pagerank_scores': None,  # Will be computed if needed
        'gene_set_matrix': None  # Will be loaded if available
    }
    
    # Save processed data
    print("Saving processed data...")
    torch.save(snp_tensor, output_dir / f'{dataset_name}_snp.pt')
    torch.save(phenotype_tensor, output_dir / f'{dataset_name}_phenotype.pt')
    
    with open(output_dir / f'{dataset_name}_graphs.pkl', 'wb') as f:
        pickle.dump(graph_data, f)
    
    # Save metadata
    metadata = {
        'n_samples': genotype.shape[0],
        'n_snps': genotype.shape[1],
        'n_traits': phenotype_df.shape[1],
        'n_genes': node_features.shape[0],
        'trait_names': list(phenotype_df.columns),
        'n_networks': len(edge_index_list)
    }
    
    with open(output_dir / f'{dataset_name}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset prepared successfully!")
    print(f"  Samples: {metadata['n_samples']}")
    print(f"  SNPs: {metadata['n_snps']}")
    print(f"  Traits: {metadata['n_traits']}")
    print(f"  Genes: {metadata['n_genes']}")
    print(f"  Networks: {metadata['n_networks']}")
    
    return metadata


def test_dataset():
    """Test PlantGPDataset with dummy data"""
    print("Testing PlantGPDataset...")
    
    # Create dummy data
    output_dir = Path('/tmp/test_dataset')
    output_dir.mkdir(exist_ok=True)
    
    n_samples = 100
    n_snps = 500
    n_traits = 3
    n_genes = 500
    
    # Dummy SNP data
    snp_data = torch.randint(0, 2, (n_samples, n_snps, 3)).float()
    torch.save(snp_data, output_dir / 'test_snp.pt')
    
    # Dummy phenotype
    phenotype = torch.randn(n_samples, n_traits)
    torch.save(phenotype, output_dir / 'test_phenotype.pt')
    
    # Dummy graph data
    node_features = torch.randn(n_genes, 64)
    edge_index = torch.randint(0, n_genes, (2, 1000))
    
    graph_data = {
        'node_features': node_features,
        'edge_index_list': [edge_index, edge_index, edge_index],
        'random_walk_features': torch.randn(n_genes, 10),
        'pagerank_scores': torch.rand(n_genes, 1),
        'gene_set_matrix': None
    }
    
    with open(output_dir / 'test_graphs.pkl', 'wb') as f:
        pickle.dump(graph_data, f)
    
    # Dummy metadata
    metadata = {
        'n_samples': n_samples,
        'n_snps': n_snps,
        'n_traits': n_traits,
        'n_genes': n_genes
    }
    
    with open(output_dir / 'test_metadata.json', 'w') as f:
        json.dump(metadata, f)
    
    # Create dataset
    dataset = PlantGPDataset(
        root=str(output_dir),
        dataset_name='test',
        split='train'
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test single sample
    sample = dataset[0]
    print(f"Sample SNP shape: {sample.snp_data.shape}")
    print(f"Sample phenotype shape: {sample.phenotype.shape}")
    print(f"Node features shape: {sample.node_features.shape}")
    print(f"Number of networks: {len(sample.edge_index_list)}")
    
    # Test DataLoader
    dataloader = PlantGPDataLoader(dataset, batch_size=16, shuffle=True)
    
    for batch_idx, (snp_batch, graph_data, pheno_batch) in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"  SNP batch shape: {snp_batch.shape}")
        print(f"  Phenotype batch shape: {pheno_batch.shape}")
        print(f"  Graph node features: {graph_data['node_features'].shape}")
        
        if batch_idx >= 2:  # Test first 3 batches
            break
    
    print("\nDataset test passed!")


if __name__ == '__main__':
    test_dataset()

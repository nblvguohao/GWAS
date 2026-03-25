#!/usr/bin/env python
"""
Build biological networks for rice469
Creates synthetic PPI, GO, and KEGG networks for testing
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch_geometric.data import Data
import json

def build_synthetic_networks(n_genes, output_dir):
    """
    Build synthetic biological networks
    In production, these would be built from STRING, GO, KEGG data
    """
    print("\n" + "="*60)
    print("Building Synthetic Biological Networks")
    print("="*60)
    print(f"Number of genes: {n_genes}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Node features (gene embeddings)
    node_features = torch.randn(n_genes, 64)
    
    networks = {}
    
    # 1. PPI Network (Protein-Protein Interaction)
    print("\n1. Building PPI Network...")
    n_edges_ppi = n_genes * 5  # Average degree of 10
    edge_index_ppi = torch.randint(0, n_genes, (2, n_edges_ppi))
    edge_weight_ppi = torch.rand(n_edges_ppi) * 0.5 + 0.5  # Weights 0.5-1.0
    
    ppi_data = Data(
        x=node_features,
        edge_index=edge_index_ppi,
        edge_attr=edge_weight_ppi,
        num_nodes=n_genes
    )
    
    torch.save(ppi_data, output_dir / "ppi_network.pt")
    print(f"  Nodes: {n_genes}")
    print(f"  Edges: {n_edges_ppi}")
    print(f"  Average degree: {n_edges_ppi * 2 / n_genes:.1f}")
    print(f"  Saved: {output_dir / 'ppi_network.pt'}")
    
    networks['ppi'] = {
        'nodes': n_genes,
        'edges': n_edges_ppi,
        'type': 'protein_protein_interaction'
    }
    
    # 2. GO Network (Gene Ontology similarity)
    print("\n2. Building GO Network...")
    n_edges_go = n_genes * 4
    edge_index_go = torch.randint(0, n_genes, (2, n_edges_go))
    edge_weight_go = torch.rand(n_edges_go) * 0.4 + 0.6  # Weights 0.6-1.0
    
    go_data = Data(
        x=node_features,
        edge_index=edge_index_go,
        edge_attr=edge_weight_go,
        num_nodes=n_genes
    )
    
    torch.save(go_data, output_dir / "go_network.pt")
    print(f"  Nodes: {n_genes}")
    print(f"  Edges: {n_edges_go}")
    print(f"  Average degree: {n_edges_go * 2 / n_genes:.1f}")
    print(f"  Saved: {output_dir / 'go_network.pt'}")
    
    networks['go'] = {
        'nodes': n_genes,
        'edges': n_edges_go,
        'type': 'gene_ontology_similarity'
    }
    
    # 3. KEGG Network (Pathway co-occurrence)
    print("\n3. Building KEGG Network...")
    n_edges_kegg = n_genes * 3
    edge_index_kegg = torch.randint(0, n_genes, (2, n_edges_kegg))
    edge_weight_kegg = torch.rand(n_edges_kegg) * 0.3 + 0.7  # Weights 0.7-1.0
    
    kegg_data = Data(
        x=node_features,
        edge_index=edge_index_kegg,
        edge_attr=edge_weight_kegg,
        num_nodes=n_genes
    )
    
    torch.save(kegg_data, output_dir / "kegg_network.pt")
    print(f"  Nodes: {n_genes}")
    print(f"  Edges: {n_edges_kegg}")
    print(f"  Average degree: {n_edges_kegg * 2 / n_genes:.1f}")
    print(f"  Saved: {output_dir / 'kegg_network.pt'}")
    
    networks['kegg'] = {
        'nodes': n_genes,
        'edges': n_edges_kegg,
        'type': 'pathway_cooccurrence'
    }
    
    # 4. Additional features
    print("\n4. Creating additional features...")
    
    # Random walk features
    random_walk_features = torch.randn(n_genes, 10)
    torch.save(random_walk_features, output_dir / "random_walk_features.pt")
    print(f"  Random walk features: {random_walk_features.shape}")
    
    # PageRank scores
    pagerank_scores = torch.rand(n_genes, 1)
    pagerank_scores = pagerank_scores / pagerank_scores.sum()
    torch.save(pagerank_scores, output_dir / "pagerank_scores.pt")
    print(f"  PageRank scores: {pagerank_scores.shape}")
    
    # Gene set matrix (100 gene sets)
    n_gene_sets = 100
    gene_set_matrix = (torch.rand(n_genes, n_gene_sets) > 0.9).float()
    torch.save(gene_set_matrix, output_dir / "gene_set_matrix.pt")
    print(f"  Gene set matrix: {gene_set_matrix.shape}")
    print(f"  Average genes per set: {gene_set_matrix.sum(0).mean():.1f}")
    
    # Save metadata
    metadata = {
        'n_genes': n_genes,
        'n_gene_sets': n_gene_sets,
        'networks': networks,
        'features': {
            'node_features_dim': 64,
            'random_walk_dim': 10,
            'pagerank_dim': 1
        }
    }
    
    with open(output_dir / "network_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n  Saved: {output_dir / 'network_metadata.json'}")
    
    return metadata


def main():
    print("="*60)
    print("Network Building Script")
    print("="*60)
    
    # Load preprocessing metadata to get n_genes
    metadata_file = Path("data/processed/rice469/metadata.json")
    
    if not metadata_file.exists():
        print(f"Error: {metadata_file} not found!")
        print("Please run preprocessing first: python scripts/preprocess_rice469.py")
        return 1
    
    with open(metadata_file, 'r') as f:
        preprocess_metadata = json.load(f)
    
    n_snps = preprocess_metadata['n_snps']
    n_genes = n_snps  # Assume 1 SNP per gene for simplicity
    
    print(f"\nBuilding networks for {n_genes} genes...")
    
    # Build networks
    output_dir = Path("data/processed/rice469/networks")
    network_metadata = build_synthetic_networks(n_genes, output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("Network Building Summary")
    print("="*60)
    print(f"Genes: {network_metadata['n_genes']}")
    print(f"Gene sets: {network_metadata['n_gene_sets']}")
    print(f"Networks built: {len(network_metadata['networks'])}")
    
    for net_name, net_info in network_metadata['networks'].items():
        print(f"\n{net_name.upper()}:")
        print(f"  Type: {net_info['type']}")
        print(f"  Edges: {net_info['edges']}")
        print(f"  Avg degree: {net_info['edges'] * 2 / net_info['nodes']:.1f}")
    
    print(f"\nNetworks saved to: {output_dir}")
    
    print("\n" + "="*60)
    print("Next Steps")
    print("="*60)
    print("1. Create data splits: python scripts/create_splits.py")
    print("2. Train model: python scripts/train_rice469.py")
    
    return 0


if __name__ == '__main__':
    exit(main())

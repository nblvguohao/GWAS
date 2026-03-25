"""
Biological network construction for PlantHGNN
Builds PPI, GO, KEGG, and heterogeneous GTM networks
"""

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
import torch
from torch_geometric.data import HeteroData, Data
from scipy.spatial.distance import pdist, squareform
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkBuilder:
    """
    Build biological networks for PlantHGNN
    - PPI network (from STRING)
    - GO functional similarity network
    - KEGG pathway co-occurrence network
    - Heterogeneous GTM (Gene-TF-Metabolite) network
    """
    
    def __init__(self, data_dir="data/raw/networks"):
        self.data_dir = Path(data_dir)
        self.networks = {}
    
    def load_string_network(self, species, score_threshold=700):
        """
        Load STRING protein-protein interaction network
        
        Args:
            species: Species name (e.g., 'oryza_sativa')
            score_threshold: Minimum combined score (0-1000)
        
        Returns:
            NetworkX graph
        """
        logger.info(f"Loading STRING network for {species} (threshold: {score_threshold})")
        
        string_file = self.data_dir / f"{species}_string_v12.txt.gz"
        
        if not string_file.exists():
            logger.warning(f"STRING file not found: {string_file}")
            logger.info("Creating mock PPI network for testing...")
            return self._create_mock_ppi_network()
        
        # Load STRING data
        df = pd.read_csv(string_file, sep=' ', compression='gzip')
        
        # Filter by score
        df = df[df['combined_score'] >= score_threshold]
        
        # Create graph
        G = nx.Graph()
        for _, row in df.iterrows():
            protein1 = row['protein1'].split('.')[-1]  # Remove species prefix
            protein2 = row['protein2'].split('.')[-1]
            score = row['combined_score'] / 1000.0  # Normalize to [0, 1]
            G.add_edge(protein1, protein2, weight=score)
        
        logger.info(f"Loaded PPI network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def _create_mock_ppi_network(self, n_genes=1000, edge_prob=0.01):
        """Create a mock PPI network for testing"""
        logger.info(f"Creating mock PPI network: {n_genes} genes")
        G = nx.erdos_renyi_graph(n_genes, edge_prob)
        
        # Relabel nodes as gene IDs
        mapping = {i: f"GENE{i:05d}" for i in range(n_genes)}
        G = nx.relabel_nodes(G, mapping)
        
        # Add random weights
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.uniform(0.7, 1.0)
        
        return G
    
    def build_go_similarity_network(self, go_annotations, similarity_threshold=0.8):
        """
        Build GO functional similarity network
        
        Args:
            go_annotations: Dict mapping gene_id to list of GO terms
            similarity_threshold: Minimum similarity to create edge
        
        Returns:
            NetworkX graph
        """
        logger.info("Building GO similarity network...")
        
        genes = list(go_annotations.keys())
        n_genes = len(genes)
        
        # Compute pairwise GO similarity (Jaccard)
        G = nx.Graph()
        G.add_nodes_from(genes)
        
        for i in tqdm(range(n_genes), desc="Computing GO similarity"):
            for j in range(i + 1, n_genes):
                gene1, gene2 = genes[i], genes[j]
                go1 = set(go_annotations[gene1])
                go2 = set(go_annotations[gene2])
                
                if len(go1) == 0 or len(go2) == 0:
                    continue
                
                # Jaccard similarity
                similarity = len(go1 & go2) / len(go1 | go2)
                
                if similarity >= similarity_threshold:
                    G.add_edge(gene1, gene2, weight=similarity)
        
        logger.info(f"GO network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def build_kegg_pathway_network(self, pathway_annotations, weight_threshold=0.1):
        """
        Build KEGG pathway co-occurrence network
        
        Args:
            pathway_annotations: Dict mapping gene_id to list of pathway IDs
            weight_threshold: Minimum Jaccard coefficient
        
        Returns:
            NetworkX graph
        """
        logger.info("Building KEGG pathway network...")
        
        genes = list(pathway_annotations.keys())
        n_genes = len(genes)
        
        G = nx.Graph()
        G.add_nodes_from(genes)
        
        for i in tqdm(range(n_genes), desc="Computing pathway similarity"):
            for j in range(i + 1, n_genes):
                gene1, gene2 = genes[i], genes[j]
                pathways1 = set(pathway_annotations[gene1])
                pathways2 = set(pathway_annotations[gene2])
                
                if len(pathways1) == 0 or len(pathways2) == 0:
                    continue
                
                # Jaccard coefficient
                jaccard = len(pathways1 & pathways2) / len(pathways1 | pathways2)
                
                if jaccard >= weight_threshold:
                    G.add_edge(gene1, gene2, weight=jaccard)
        
        logger.info(f"KEGG network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def build_heterogeneous_gtm_network(self, ppi_graph, tf_targets, metabolite_genes):
        """
        Build heterogeneous Gene-TF-Metabolite (GTM) network
        
        Args:
            ppi_graph: Gene-gene PPI network
            tf_targets: Dict mapping TF to list of target genes
            metabolite_genes: Dict mapping metabolite to list of associated genes
        
        Returns:
            PyG HeteroData object
        """
        logger.info("Building heterogeneous GTM network...")
        
        data = HeteroData()
        
        # Node types: gene, tf, metabolite
        genes = list(ppi_graph.nodes())
        tfs = list(tf_targets.keys())
        metabolites = list(metabolite_genes.keys())
        
        # Create node mappings
        gene_to_idx = {g: i for i, g in enumerate(genes)}
        tf_to_idx = {t: i for i, t in enumerate(tfs)}
        metabolite_to_idx = {m: i for i, m in enumerate(metabolites)}
        
        # Add node features (placeholder - will be filled with actual features)
        data['gene'].num_nodes = len(genes)
        data['tf'].num_nodes = len(tfs)
        data['metabolite'].num_nodes = len(metabolites)
        
        # Edge type 1: gene-gene (from PPI)
        gene_edges = []
        for u, v in ppi_graph.edges():
            if u in gene_to_idx and v in gene_to_idx:
                gene_edges.append([gene_to_idx[u], gene_to_idx[v]])
        
        if gene_edges:
            edge_index = torch.tensor(gene_edges, dtype=torch.long).t()
            data['gene', 'interacts', 'gene'].edge_index = edge_index
        
        # Edge type 2: tf-gene (regulatory)
        tf_gene_edges = []
        for tf, targets in tf_targets.items():
            if tf not in tf_to_idx:
                continue
            tf_idx = tf_to_idx[tf]
            for target in targets:
                if target in gene_to_idx:
                    tf_gene_edges.append([tf_idx, gene_to_idx[target]])
        
        if tf_gene_edges:
            edge_index = torch.tensor(tf_gene_edges, dtype=torch.long).t()
            data['tf', 'regulates', 'gene'].edge_index = edge_index
        
        # Edge type 3: gene-metabolite
        gene_metabolite_edges = []
        for metabolite, genes_list in metabolite_genes.items():
            if metabolite not in metabolite_to_idx:
                continue
            met_idx = metabolite_to_idx[metabolite]
            for gene in genes_list:
                if gene in gene_to_idx:
                    gene_metabolite_edges.append([gene_to_idx[gene], met_idx])
        
        if gene_metabolite_edges:
            edge_index = torch.tensor(gene_metabolite_edges, dtype=torch.long).t()
            data['gene', 'produces', 'metabolite'].edge_index = edge_index
        
        logger.info(f"GTM network created:")
        logger.info(f"  Genes: {data['gene'].num_nodes}")
        logger.info(f"  TFs: {data['tf'].num_nodes}")
        logger.info(f"  Metabolites: {data['metabolite'].num_nodes}")
        
        return data
    
    def networkx_to_pyg(self, G, node_features=None):
        """
        Convert NetworkX graph to PyTorch Geometric Data object
        
        Args:
            G: NetworkX graph
            node_features: Optional node feature matrix (n_nodes × d)
        
        Returns:
            PyG Data object
        """
        # Create node mapping
        nodes = list(G.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        
        # Edge index
        edge_list = []
        edge_weights = []
        for u, v, data in G.edges(data=True):
            edge_list.append([node_to_idx[u], node_to_idx[v]])
            edge_list.append([node_to_idx[v], node_to_idx[u]])  # Undirected
            weight = data.get('weight', 1.0)
            edge_weights.extend([weight, weight])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)
        
        # Node features
        if node_features is None:
            x = torch.eye(len(nodes))  # One-hot encoding
        else:
            x = torch.tensor(node_features, dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
        
        return data, nodes
    
    def save_networks(self, output_dir, dataset_name):
        """Save all networks to disk"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for network_name, network_data in self.networks.items():
            output_path = output_dir / f"{dataset_name}_{network_name}.pt"
            torch.save(network_data, output_path)
            logger.info(f"Saved {network_name} to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Build biological networks for PlantHGNN')
    parser.add_argument('--species', required=True, 
                       choices=['oryza_sativa', 'zea_mays', 'glycine_max', 'triticum_aestivum'],
                       help='Plant species')
    parser.add_argument('--output-dir', default='data/processed/graphs',
                       help='Output directory')
    parser.add_argument('--string-threshold', type=int, default=700,
                       help='STRING combined score threshold')
    
    args = parser.parse_args()
    
    builder = NetworkBuilder()
    
    # Build PPI network
    ppi_graph = builder.load_string_network(args.species, args.string_threshold)
    ppi_data, nodes = builder.networkx_to_pyg(ppi_graph)
    
    builder.networks['ppi'] = ppi_data
    
    # Save networks
    builder.save_networks(args.output_dir, args.species)
    
    logger.info("Network construction complete!")


if __name__ == '__main__':
    main()

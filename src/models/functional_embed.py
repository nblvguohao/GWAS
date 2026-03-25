"""
Functional embedding module for PlantHGNN
Based on GRAFT's functional embedding approach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FunctionalEmbedding(nn.Module):
    """
    Functional embedding module
    
    Embeds genes based on their membership in functional gene sets
    (GO terms, KEGG pathways, PlantCyc pathways, etc.)
    
    Args:
        n_genes: Number of genes
        n_gene_sets: Number of functional gene sets
        embedding_dim: Embedding dimension
        gene_set_matrix: Binary matrix (n_genes, n_gene_sets) indicating membership
    """
    
    def __init__(self, n_genes, n_gene_sets, embedding_dim, gene_set_matrix=None):
        super().__init__()
        
        self.n_genes = n_genes
        self.n_gene_sets = n_gene_sets
        self.embedding_dim = embedding_dim
        
        # Gene set membership matrix
        if gene_set_matrix is not None:
            self.register_buffer('gene_set_matrix', torch.tensor(gene_set_matrix, dtype=torch.float))
        else:
            # Random initialization for testing
            self.register_buffer('gene_set_matrix', torch.rand(n_genes, n_gene_sets) > 0.9)
        
        # Learnable gene set embeddings
        self.gene_set_embeddings = nn.Parameter(torch.randn(n_gene_sets, embedding_dim))
        
        # Projection layer
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.gene_set_embeddings)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
    
    def forward(self, gene_indices=None):
        """
        Forward pass
        
        Args:
            gene_indices: Optional tensor of gene indices (batch_size,)
                         If None, returns embeddings for all genes
        
        Returns:
            Functional embeddings (batch_size, embedding_dim) or (n_genes, embedding_dim)
        """
        # Compute gene embeddings as weighted sum of gene set embeddings
        # Shape: (n_genes, embedding_dim)
        gene_embeddings = torch.matmul(self.gene_set_matrix.float(), self.gene_set_embeddings)
        
        # Normalize by number of gene sets per gene
        gene_set_counts = self.gene_set_matrix.sum(dim=1, keepdim=True).clamp(min=1)
        gene_embeddings = gene_embeddings / gene_set_counts
        
        # Project
        gene_embeddings = self.projection(gene_embeddings)
        gene_embeddings = self.layer_norm(gene_embeddings)
        
        # Select specific genes if indices provided
        if gene_indices is not None:
            gene_embeddings = gene_embeddings[gene_indices]
        
        return gene_embeddings
    
    def get_gene_set_importance(self, gene_idx):
        """
        Get importance of each gene set for a specific gene
        Useful for interpretability
        
        Args:
            gene_idx: Gene index
        
        Returns:
            Gene set importance scores
        """
        membership = self.gene_set_matrix[gene_idx]
        return membership.detach()


class StructuralEncoder(nn.Module):
    """
    Structural encoding module
    
    Encodes graph structural information:
    - Random walk positional encoding (local topology)
    - PageRank centrality (global importance)
    
    Args:
        d_model: Model dimension
        max_walk_length: Maximum random walk length
    """
    
    def __init__(self, d_model, max_walk_length=10):
        super().__init__()
        
        self.d_model = d_model
        self.max_walk_length = max_walk_length
        
        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(max_walk_length, d_model))
        
        # Centrality projection
        self.centrality_proj = nn.Linear(1, d_model)
        
        # Fusion layer
        self.fusion = nn.Linear(d_model * 2, d_model)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.pos_encoding)
        nn.init.xavier_uniform_(self.centrality_proj.weight)
        nn.init.zeros_(self.centrality_proj.bias)
        nn.init.xavier_uniform_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)
    
    def forward(self, random_walk_features, pagerank_scores):
        """
        Forward pass
        
        Args:
            random_walk_features: Random walk features (n_nodes, max_walk_length)
            pagerank_scores: PageRank centrality scores (n_nodes, 1)
        
        Returns:
            Structural embeddings (n_nodes, d_model)
        """
        # Random walk positional encoding
        # Shape: (n_nodes, d_model)
        pos_embed = torch.matmul(random_walk_features, self.pos_encoding)
        
        # PageRank centrality encoding
        # Shape: (n_nodes, d_model)
        centrality_embed = self.centrality_proj(pagerank_scores)
        
        # Concatenate and fuse
        combined = torch.cat([pos_embed, centrality_embed], dim=-1)
        structural_embed = self.fusion(combined)
        
        return F.relu(structural_embed)
    
    @staticmethod
    def compute_random_walk_features(edge_index, n_nodes, max_walk_length=10, n_walks=100):
        """
        Compute random walk features for nodes
        
        Args:
            edge_index: Edge index tensor (2, n_edges)
            n_nodes: Number of nodes
            max_walk_length: Maximum walk length
            n_walks: Number of random walks per node
        
        Returns:
            Random walk feature matrix (n_nodes, max_walk_length)
        """
        # Build adjacency list
        adj_list = [[] for _ in range(n_nodes)]
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            adj_list[src].append(dst)
        
        # Perform random walks
        walk_counts = torch.zeros(n_nodes, max_walk_length)
        
        for node in range(n_nodes):
            for _ in range(n_walks):
                current = node
                for step in range(max_walk_length):
                    if len(adj_list[current]) == 0:
                        break
                    # Random neighbor
                    current = np.random.choice(adj_list[current])
                    walk_counts[node, step] += 1
        
        # Normalize
        walk_counts = walk_counts / (n_walks + 1e-8)
        
        return walk_counts
    
    @staticmethod
    def compute_pagerank(edge_index, n_nodes, alpha=0.85, max_iter=100):
        """
        Compute PageRank scores
        
        Args:
            edge_index: Edge index tensor (2, n_edges)
            n_nodes: Number of nodes
            alpha: Damping factor
            max_iter: Maximum iterations
        
        Returns:
            PageRank scores (n_nodes, 1)
        """
        # Build adjacency matrix
        adj = torch.zeros(n_nodes, n_nodes)
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            adj[src, dst] = 1
        
        # Normalize by out-degree
        out_degree = adj.sum(dim=1, keepdim=True).clamp(min=1)
        adj_norm = adj / out_degree
        
        # Power iteration
        pr = torch.ones(n_nodes, 1) / n_nodes
        
        for _ in range(max_iter):
            pr_new = alpha * torch.matmul(adj_norm.t(), pr) + (1 - alpha) / n_nodes
            if torch.norm(pr_new - pr) < 1e-6:
                break
            pr = pr_new
        
        return pr


def test_functional_embed():
    """Test functional embedding module"""
    print("Testing FunctionalEmbedding...")
    
    n_genes = 1000
    n_gene_sets = 200
    embedding_dim = 128
    
    # Create random gene set matrix
    gene_set_matrix = (torch.rand(n_genes, n_gene_sets) > 0.9).float()
    
    # Create module
    func_embed = FunctionalEmbedding(n_genes, n_gene_sets, embedding_dim, gene_set_matrix)
    
    # Forward pass (all genes)
    embeddings = func_embed()
    print(f"All genes embeddings shape: {embeddings.shape}")
    
    # Forward pass (specific genes)
    gene_indices = torch.tensor([0, 10, 100, 500])
    embeddings_subset = func_embed(gene_indices)
    print(f"Subset embeddings shape: {embeddings_subset.shape}")
    
    # Test gene set importance
    importance = func_embed.get_gene_set_importance(0)
    print(f"Gene set importance for gene 0: {importance.sum()} sets")
    
    print("\nTesting StructuralEncoder...")
    
    # Create random structural features
    random_walk_features = torch.rand(n_genes, 10)
    pagerank_scores = torch.rand(n_genes, 1)
    
    struct_encoder = StructuralEncoder(embedding_dim)
    struct_embeddings = struct_encoder(random_walk_features, pagerank_scores)
    
    print(f"Structural embeddings shape: {struct_embeddings.shape}")
    
    print("\nFunctional embedding test passed!")


if __name__ == '__main__':
    test_functional_embed()

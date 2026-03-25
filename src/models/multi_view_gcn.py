"""
Multi-view GCN encoder for PlantHGNN
Processes multiple biological networks (PPI, GO, KEGG) with attention fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import math


class MultiViewGCNEncoder(nn.Module):
    """
    Multi-view GCN encoder with attention fusion
    
    Processes multiple biological networks separately and fuses them
    using learnable attention weights.
    
    Args:
        in_dim: Input feature dimension
        hidden_dim: Hidden dimension
        out_dim: Output dimension
        n_views: Number of network views (default: 3 for PPI, GO, KEGG)
        n_layers: Number of GCN layers per view
        dropout: Dropout rate
    """
    
    def __init__(self, in_dim, hidden_dim, out_dim, n_views=3, n_layers=2, dropout=0.2):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_views = n_views
        self.n_layers = n_layers
        
        # GCN layers for each view
        self.view_encoders = nn.ModuleList()
        for _ in range(n_views):
            layers = nn.ModuleList()
            
            # First layer
            layers.append(GCNConv(in_dim, hidden_dim))
            
            # Hidden layers
            for _ in range(n_layers - 2):
                layers.append(GCNConv(hidden_dim, hidden_dim))
            
            # Output layer
            if n_layers > 1:
                layers.append(GCNConv(hidden_dim, out_dim))
            
            self.view_encoders.append(layers)
        
        # Attention fusion mechanism
        self.attention_weights = nn.Parameter(torch.ones(n_views))
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim if i < n_layers - 1 else out_dim)
            for i in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index_list, edge_weight_list=None, return_attention=False):
        """
        Forward pass
        
        Args:
            x: Node features (n_nodes, in_dim)
            edge_index_list: List of edge indices for each view
            edge_weight_list: Optional list of edge weights for each view
            return_attention: Whether to return attention weights
        
        Returns:
            Fused node embeddings (n_nodes, out_dim)
            Optional: attention weights (n_views,)
        """
        if len(edge_index_list) != self.n_views:
            raise ValueError(f"Expected {self.n_views} edge indices, got {len(edge_index_list)}")
        
        if edge_weight_list is None:
            edge_weight_list = [None] * self.n_views
        
        # Encode each view separately
        view_embeddings = []
        
        for view_idx in range(self.n_views):
            h = x
            edge_index = edge_index_list[view_idx]
            edge_weight = edge_weight_list[view_idx]
            
            # Apply GCN layers
            for layer_idx, gcn_layer in enumerate(self.view_encoders[view_idx]):
                h = gcn_layer(h, edge_index, edge_weight)
                
                # Batch normalization
                if layer_idx < len(self.batch_norms):
                    h = self.batch_norms[layer_idx](h)
                
                # Activation (except last layer)
                if layer_idx < len(self.view_encoders[view_idx]) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
            
            view_embeddings.append(h)
        
        # Stack view embeddings
        # Shape: (n_views, n_nodes, out_dim)
        stacked_embeddings = torch.stack(view_embeddings, dim=0)
        
        # Compute attention weights
        # Softmax over views
        attn_weights = F.softmax(self.attention_weights, dim=0)
        
        # Weighted fusion
        # Shape: (n_nodes, out_dim)
        fused_embedding = torch.einsum('v,vnd->nd', attn_weights, stacked_embeddings)
        
        if return_attention:
            return fused_embedding, attn_weights.detach()
        
        return fused_embedding
    
    def get_view_embeddings(self, x, edge_index_list, edge_weight_list=None):
        """
        Get embeddings from each view separately (for analysis)
        
        Returns:
            List of view embeddings
        """
        if edge_weight_list is None:
            edge_weight_list = [None] * self.n_views
        
        view_embeddings = []
        
        for view_idx in range(self.n_views):
            h = x
            edge_index = edge_index_list[view_idx]
            edge_weight = edge_weight_list[view_idx]
            
            for layer_idx, gcn_layer in enumerate(self.view_encoders[view_idx]):
                h = gcn_layer(h, edge_index, edge_weight)
                
                if layer_idx < len(self.batch_norms):
                    h = self.batch_norms[layer_idx](h)
                
                if layer_idx < len(self.view_encoders[view_idx]) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
            
            view_embeddings.append(h.detach())
        
        return view_embeddings


class SingleViewGCN(nn.Module):
    """
    Single-view GCN (for ablation study)
    Equivalent to NetGP baseline
    """
    
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=2, dropout=0.2):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(GCNConv(in_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(n_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layer
        if n_layers > 1:
            self.layers.append(GCNConv(hidden_dim, out_dim))
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim if i < n_layers - 1 else out_dim)
            for i in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_weight=None):
        """Forward pass"""
        h = x
        
        for layer_idx, gcn_layer in enumerate(self.layers):
            h = gcn_layer(h, edge_index, edge_weight)
            
            if layer_idx < len(self.batch_norms):
                h = self.batch_norms[layer_idx](h)
            
            if layer_idx < len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        
        return h


def test_multi_view_gcn():
    """Test multi-view GCN implementation"""
    print("Testing MultiViewGCNEncoder...")
    
    n_nodes = 100
    in_dim = 64
    hidden_dim = 128
    out_dim = 128
    n_views = 3
    
    # Create random node features
    x = torch.randn(n_nodes, in_dim)
    
    # Create random edge indices for each view
    edge_index_list = []
    for _ in range(n_views):
        # Random edges
        n_edges = 200
        edge_index = torch.randint(0, n_nodes, (2, n_edges))
        edge_index_list.append(edge_index)
    
    # Create model
    model = MultiViewGCNEncoder(in_dim, hidden_dim, out_dim, n_views=n_views)
    
    # Forward pass
    output, attn_weights = model(x, edge_index_list, return_attention=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights: {attn_weights}")
    print(f"Attention weights sum: {attn_weights.sum()}")
    
    # Test view embeddings
    view_embeddings = model.get_view_embeddings(x, edge_index_list)
    print(f"\nView embeddings:")
    for i, emb in enumerate(view_embeddings):
        print(f"  View {i}: {emb.shape}")
    
    print("\nMultiViewGCN test passed!")


if __name__ == '__main__':
    test_multi_view_gcn()

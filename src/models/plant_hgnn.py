"""
PlantHGNN: Main model implementation
Plant Heterogeneous Graph Neural Network with Attention Residuals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_residual import AttnResTransformer
from .multi_view_gcn import MultiViewGCNEncoder, SingleViewGCN
from .functional_embed import FunctionalEmbedding, StructuralEncoder


class SNPEncoder(nn.Module):
    """
    SNP feature encoder
    
    Encodes one-hot SNP features into continuous embeddings
    
    Args:
        n_snps: Number of SNPs
        snp_encoding_dim: Encoding dimension per SNP (3 for one-hot)
        d_model: Output dimension
    """
    
    def __init__(self, n_snps, snp_encoding_dim=3, d_model=128):
        super().__init__()
        
        self.n_snps = n_snps
        self.snp_encoding_dim = snp_encoding_dim
        self.d_model = d_model
        
        # Convolutional encoder for SNP sequences
        self.conv1 = nn.Conv1d(snp_encoding_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Projection to d_model
        self.projection = nn.Linear(128, d_model)
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, snp_data):
        """
        Forward pass
        
        Args:
            snp_data: SNP matrix (batch_size, n_snps, snp_encoding_dim)
        
        Returns:
            SNP embeddings (batch_size, d_model)
        """
        # Transpose for conv1d: (batch_size, snp_encoding_dim, n_snps)
        x = snp_data.transpose(1, 2)
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        
        # Global pooling
        x = self.pool(x).squeeze(-1)  # (batch_size, 128)
        
        # Project to d_model
        x = self.projection(x)
        
        return x


class PlantHGNN(nn.Module):
    """
    PlantHGNN: Plant Heterogeneous Graph Neural Network
    
    Main model combining:
    - SNP encoding
    - Multi-view GCN (PPI, GO, KEGG networks)
    - Functional embedding
    - Structural encoding
    - AttnRes Transformer
    - Regression head for GEBV prediction
    
    Args:
        n_snps: Number of SNPs
        n_genes: Number of genes
        n_gene_sets: Number of functional gene sets
        n_traits: Number of traits to predict
        d_model: Hidden dimension (default: 128)
        n_transformer_layers: Number of transformer layers (default: 6)
        n_attnres_blocks: Number of AttnRes blocks (default: 8)
        n_gcn_layers: Number of GCN layers (default: 2)
        n_views: Number of network views (default: 3)
        dropout: Dropout rate (default: 0.2)
        use_heterogeneous: Whether to use heterogeneous network (default: True)
        use_attnres: Whether to use AttnRes (default: True)
        use_functional_embed: Whether to use functional embedding (default: True)
        use_structural_encode: Whether to use structural encoding (default: True)
    """
    
    def __init__(
        self,
        n_snps,
        n_genes,
        n_gene_sets,
        n_traits,
        d_model=128,
        n_transformer_layers=6,
        n_attnres_blocks=8,
        n_gcn_layers=2,
        n_views=3,
        dropout=0.2,
        use_heterogeneous=True,
        use_attnres=True,
        use_functional_embed=True,
        use_structural_encode=True,
        gene_set_matrix=None
    ):
        super().__init__()
        
        self.n_snps = n_snps
        self.n_genes = n_genes
        self.n_traits = n_traits
        self.d_model = d_model
        self.use_heterogeneous = use_heterogeneous
        self.use_attnres = use_attnres
        self.use_functional_embed = use_functional_embed
        self.use_structural_encode = use_structural_encode
        
        # SNP encoder
        self.snp_encoder = SNPEncoder(n_snps, snp_encoding_dim=3, d_model=d_model)
        
        # Node feature projection (to handle variable input dimensions)
        self.node_projection = nn.Linear(64, d_model)  # Project node features to d_model
        
        # Multi-view GCN encoder
        if n_views > 1:
            self.multi_view_gcn = MultiViewGCNEncoder(
                in_dim=d_model,
                hidden_dim=d_model,
                out_dim=d_model,
                n_views=n_views,
                n_layers=n_gcn_layers,
                dropout=dropout
            )
        else:
            # Single view (for ablation)
            self.multi_view_gcn = SingleViewGCN(
                in_dim=d_model,
                hidden_dim=d_model,
                out_dim=d_model,
                n_layers=n_gcn_layers,
                dropout=dropout
            )
        
        # Functional embedding
        if use_functional_embed:
            self.functional_embed = FunctionalEmbedding(
                n_genes, n_gene_sets, d_model, gene_set_matrix
            )
        
        # Structural encoder
        if use_structural_encode:
            self.structural_encoder = StructuralEncoder(d_model)
        
        # Calculate input dimension for transformer
        transformer_input_dim = d_model  # SNP features
        transformer_input_dim += d_model  # GCN features
        if use_functional_embed:
            transformer_input_dim += d_model
        if use_structural_encode:
            transformer_input_dim += d_model
        
        # Input projection
        self.input_projection = nn.Linear(transformer_input_dim, d_model)
        
        # Transformer encoder with AttnRes
        if use_attnres:
            self.transformer = AttnResTransformer(
                d_model=d_model,
                n_layers=n_transformer_layers,
                n_blocks=n_attnres_blocks,
                dropout=dropout
            )
        else:
            # Standard transformer (for ablation)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, n_transformer_layers)
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_traits)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, snp_data, graph_data, return_attention=False):
        """
        Forward pass
        
        Args:
            snp_data: SNP matrix (batch_size, n_snps, 3)
            graph_data: Dict containing:
                - 'node_features': Node features (n_genes, d_model)
                - 'edge_index_list': List of edge indices for each view
                - 'edge_weight_list': Optional edge weights
                - 'random_walk_features': Random walk features (n_genes, max_walk_length)
                - 'pagerank_scores': PageRank scores (n_genes, 1)
            return_attention: Whether to return attention weights
        
        Returns:
            Predictions (batch_size, n_traits)
            Optional: attention weights dict
        """
        batch_size = snp_data.shape[0]
        
        # 1. Encode SNPs
        snp_embed = self.snp_encoder(snp_data)  # (batch_size, d_model)
        
        # 2. Multi-view GCN encoding
        node_features = graph_data['node_features']
        edge_index_list = graph_data['edge_index_list']
        edge_weight_list = graph_data.get('edge_weight_list', None)
        
        # Project node features to d_model dimension
        node_features_projected = self.node_projection(node_features)
        
        if isinstance(self.multi_view_gcn, MultiViewGCNEncoder):
            gcn_embed, network_attn = self.multi_view_gcn(
                node_features_projected, edge_index_list, edge_weight_list, return_attention=True
            )
        else:
            # Single view
            gcn_embed = self.multi_view_gcn(node_features, edge_index_list[0], 
                                           edge_weight_list[0] if edge_weight_list else None)
            network_attn = None
        
        # Pool GCN embeddings to batch size (mean pooling)
        gcn_embed_pooled = gcn_embed.mean(dim=0, keepdim=True).expand(batch_size, -1)
        
        # 3. Functional embedding
        feature_list = [snp_embed, gcn_embed_pooled]
        
        if self.use_functional_embed:
            func_embed = self.functional_embed()  # (n_genes, d_model)
            func_embed_pooled = func_embed.mean(dim=0, keepdim=True).expand(batch_size, -1)
            feature_list.append(func_embed_pooled)
        
        # 4. Structural encoding
        if self.use_structural_encode:
            random_walk_features = graph_data['random_walk_features']
            pagerank_scores = graph_data['pagerank_scores']
            struct_embed = self.structural_encoder(random_walk_features, pagerank_scores)
            struct_embed_pooled = struct_embed.mean(dim=0, keepdim=True).expand(batch_size, -1)
            feature_list.append(struct_embed_pooled)
        
        # 5. Concatenate all features
        h_input = torch.cat(feature_list, dim=-1)  # (batch_size, total_dim)
        
        # 6. Project to d_model
        h = self.input_projection(h_input)  # (batch_size, d_model)
        h = h.unsqueeze(1)  # (batch_size, 1, d_model) for transformer
        
        # 7. Transformer encoding
        if self.use_attnres:
            h, attn_info = self.transformer(h, return_attention_weights=True)
        else:
            h = self.transformer(h)
            attn_info = None
        
        h = h.squeeze(1)  # (batch_size, d_model)
        
        # 8. Regression head
        predictions = self.regression_head(h)  # (batch_size, n_traits)
        
        if return_attention:
            attention_dict = {
                'network_attention': network_attn,
                'transformer_attention': attn_info
            }
            return predictions, attention_dict
        
        return predictions
    
    def get_network_attention_weights(self):
        """Get multi-view network attention weights"""
        if isinstance(self.multi_view_gcn, MultiViewGCNEncoder):
            return F.softmax(self.multi_view_gcn.attention_weights, dim=0).detach()
        return None
    
    def get_depth_attention_weights(self):
        """Get AttnRes depth attention weights"""
        if self.use_attnres:
            return self.transformer.get_depth_attention_weights()
        return None


def test_plant_hgnn():
    """Test PlantHGNN model"""
    print("Testing PlantHGNN...")
    
    batch_size = 4
    n_snps = 1000
    n_genes = 500
    n_gene_sets = 100
    n_traits = 3
    d_model = 128
    
    # Create dummy data
    snp_data = torch.randn(batch_size, n_snps, 3)
    
    # Create dummy graph data
    node_features = torch.randn(n_genes, d_model)
    edge_index_list = [
        torch.randint(0, n_genes, (2, 1000)),
        torch.randint(0, n_genes, (2, 1000)),
        torch.randint(0, n_genes, (2, 1000))
    ]
    random_walk_features = torch.rand(n_genes, 10)
    pagerank_scores = torch.rand(n_genes, 1)
    
    graph_data = {
        'node_features': node_features,
        'edge_index_list': edge_index_list,
        'random_walk_features': random_walk_features,
        'pagerank_scores': pagerank_scores
    }
    
    # Create model (fix: n_layers must be divisible by n_blocks)
    model = PlantHGNN(
        n_snps=n_snps,
        n_genes=n_genes,
        n_gene_sets=n_gene_sets,
        n_traits=n_traits,
        d_model=d_model,
        n_transformer_layers=8,  # Must be divisible by n_attnres_blocks (8)
        n_attnres_blocks=8
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    predictions, attention_dict = model(snp_data, graph_data, return_attention=True)
    
    print(f"Input SNP shape: {snp_data.shape}")
    print(f"Output predictions shape: {predictions.shape}")
    print(f"Network attention: {attention_dict['network_attention']}")
    
    # Test attention weight extraction
    network_attn = model.get_network_attention_weights()
    depth_attn = model.get_depth_attention_weights()
    
    print(f"Network attention weights: {network_attn}")
    print(f"Depth attention weights shape: {depth_attn.shape if depth_attn is not None else None}")
    
    print("\nPlantHGNN test passed!")


if __name__ == '__main__':
    test_plant_hgnn()

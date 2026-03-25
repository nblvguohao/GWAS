"""
Attention Residuals (AttnRes) implementation for PlantHGNN
Based on Kimi's Attention Residuals (arxiv 2603.15031)
Reference: https://github.com/MoonshotAI/Attention-Residuals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BlockAttnRes(nn.Module):
    """
    Block Attention Residuals
    
    Aggregates hidden states from previous layers using learned attention weights.
    Key difference from standard residuals: query vectors are trainable parameters,
    not dependent on current input.
    
    Formula:
        h_l = Σ_{i=0}^{l-1} α_{i→l} · v_i
        where α = softmax(q_l K^T / sqrt(d))
        q_l is a trainable parameter vector for layer l
    
    Args:
        d_model: Hidden dimension
        n_blocks: Number of blocks (default: 8, following Kimi paper)
        dropout: Dropout rate
    """
    
    def __init__(self, d_model, n_blocks=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.dropout = dropout
        
        # Trainable query vectors for each block
        # Shape: (n_blocks, d_model)
        self.query_vectors = nn.Parameter(torch.randn(n_blocks, d_model))
        
        # Key projection
        self.key_proj = nn.Linear(d_model, d_model)
        
        # Value projection
        self.value_proj = nn.Linear(d_model, d_model)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Initialize parameters
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.query_vectors)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.key_proj.bias)
        nn.init.zeros_(self.value_proj.bias)
    
    def forward(self, layer_outputs, current_block_idx):
        """
        Forward pass
        
        Args:
            layer_outputs: List of hidden states from previous layers
                          Each element has shape (batch_size, seq_len, d_model)
            current_block_idx: Index of current block (0 to n_blocks-1)
        
        Returns:
            Aggregated hidden state (batch_size, seq_len, d_model)
            Attention weights (n_previous_layers,)
        """
        if len(layer_outputs) == 0:
            raise ValueError("layer_outputs cannot be empty")
        
        # Stack previous layer outputs
        # Shape: (n_layers, batch_size, seq_len, d_model)
        stacked_outputs = torch.stack(layer_outputs, dim=0)
        n_layers, batch_size, seq_len, d_model = stacked_outputs.shape
        
        # Get query vector for current block
        # Shape: (d_model,)
        query = self.query_vectors[current_block_idx]
        
        # Project keys and values
        # Shape: (n_layers, batch_size, seq_len, d_model)
        keys = self.key_proj(stacked_outputs)
        values = self.value_proj(stacked_outputs)
        
        # Compute attention scores
        # Average over sequence length to get layer-level representations
        # Shape: (n_layers, batch_size, d_model)
        keys_pooled = keys.mean(dim=2)
        
        # Compute attention: query @ keys^T
        # Shape: (n_layers, batch_size)
        scores = torch.einsum('d,lbd->lb', query, keys_pooled) / math.sqrt(d_model)
        
        # Softmax over layers
        # Shape: (n_layers, batch_size)
        attn_weights = F.softmax(scores, dim=0)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Weighted sum of values
        # Shape: (batch_size, seq_len, d_model)
        attn_weights_expanded = attn_weights.unsqueeze(2).unsqueeze(3)  # (n_layers, batch_size, 1, 1)
        aggregated = (values * attn_weights_expanded).sum(dim=0)
        
        # Layer normalization
        aggregated = self.layer_norm(aggregated)
        
        # Return aggregated output and attention weights (averaged over batch)
        attn_weights_avg = attn_weights.mean(dim=1)  # (n_layers,)
        
        return aggregated, attn_weights_avg
    
    def get_attention_weights(self):
        """
        Get current attention weight distribution
        Useful for interpretability analysis
        """
        return self.query_vectors.detach()


class TransformerLayerWithAttnRes(nn.Module):
    """
    Transformer layer with Attention Residuals
    
    Combines standard transformer operations with AttnRes aggregation
    """
    
    def __init__(self, d_model, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class AttnResTransformer(nn.Module):
    """
    Complete Transformer encoder with Attention Residuals
    
    Args:
        d_model: Hidden dimension
        n_layers: Number of transformer layers
        n_blocks: Number of AttnRes blocks (should divide n_layers evenly)
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
    """
    
    def __init__(self, d_model, n_layers=6, n_blocks=8, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_blocks = n_blocks
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayerWithAttnRes(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Attention Residuals module
        self.attn_res = BlockAttnRes(d_model, n_blocks, dropout)
        
        # Determine layers per block
        self.layers_per_block = n_layers // n_blocks
        if self.layers_per_block * n_blocks != n_layers:
            raise ValueError(f"n_layers ({n_layers}) must be divisible by n_blocks ({n_blocks})")
    
    def forward(self, x, mask=None, return_attention_weights=False):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
            return_attention_weights: Whether to return AttnRes attention weights
        
        Returns:
            Output tensor (batch_size, seq_len, d_model)
            Optional: attention weights dict
        """
        layer_outputs = [x]  # Store outputs from all layers
        all_attn_weights = []
        
        for layer_idx, layer in enumerate(self.layers):
            # Standard transformer layer
            x = layer(x, mask)
            
            # Apply AttnRes at block boundaries
            block_idx = layer_idx // self.layers_per_block
            
            if (layer_idx + 1) % self.layers_per_block == 0 and layer_idx < self.n_layers - 1:
                # Apply attention residuals
                x, attn_weights = self.attn_res(layer_outputs, block_idx)
                all_attn_weights.append(attn_weights)
            
            layer_outputs.append(x)
        
        if return_attention_weights:
            return x, {'attn_res_weights': all_attn_weights}
        
        return x
    
    def get_depth_attention_weights(self):
        """
        Get attention weights for interpretability analysis
        
        Returns:
            Tensor of shape (n_blocks, d_model) containing query vectors
        """
        return self.attn_res.get_attention_weights()


def test_attn_res():
    """Test AttnRes implementation"""
    print("Testing BlockAttnRes...")
    
    batch_size = 4
    seq_len = 10
    d_model = 128
    n_blocks = 8
    
    # Create module
    attn_res = BlockAttnRes(d_model, n_blocks)
    
    # Create dummy layer outputs
    layer_outputs = [torch.randn(batch_size, seq_len, d_model) for _ in range(5)]
    
    # Forward pass
    output, attn_weights = attn_res(layer_outputs, current_block_idx=0)
    
    print(f"Input: {len(layer_outputs)} layers of shape {layer_outputs[0].shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Attention weights: {attn_weights}")
    
    print("\nTesting AttnResTransformer...")
    
    # Create transformer
    transformer = AttnResTransformer(d_model=128, n_layers=8, n_blocks=8)
    
    # Forward pass
    x = torch.randn(batch_size, seq_len, d_model)
    output, attn_info = transformer(x, return_attention_weights=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of AttnRes applications: {len(attn_info['attn_res_weights'])}")
    
    print("\nAttnRes test passed!")


if __name__ == '__main__':
    test_attn_res()

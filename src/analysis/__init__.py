"""Analysis and interpretability modules for PlantHGNN"""

from .interpretability import analyze_depth_attention
from .network_contrib import analyze_network_contribution
from .visualization import plot_gene_embedding_umap

__all__ = ['analyze_depth_attention', 'analyze_network_contribution', 'plot_gene_embedding_umap']

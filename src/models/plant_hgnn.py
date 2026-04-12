"""
PlantHGNN: 植物基因组预测异构图神经网络

架构（消融实验可选各模块）：
  1. SNP编码器: Linear + BN → d_model
  2. (可选) MultiViewGCN: 多视图图网络编码基因嵌入
  3. Transformer + BlockAttnRes: 深度特征提取
  4. 回归头: Linear → n_traits

本文件是实验性实现，支持通过 use_* 开关进行消融实验。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .attention_residual import BlockAttnRes
from .multi_view_gcn import MultiViewGCNEncoder


# ══════════════════════════════════════════════════════════════════════════════
# 辅助模块
# ══════════════════════════════════════════════════════════════════════════════

class SNPEncoder(nn.Module):
    """
    将 (batch, n_snps) 的标准化SNP矩阵编码为 (batch, d_model)。
    使用带残差的两层MLP + BN。
    """
    def __init__(self, n_snps: int, d_model: int, dropout: float = 0.2):
        super().__init__()
        hidden = min(n_snps, d_model * 4)
        self.net = nn.Sequential(
            nn.Linear(n_snps, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.BatchNorm1d(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_snps) → (batch, d_model)"""
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    标准 Pre-Norm Transformer block（用于序列长度=1的情况，
    即每个样本是一个d维向量时的特征变换）。
    使用 Self-Attention 在 batch 内部不做跨样本交互，
    而是做特征维度内的非线性变换（FFN-style）。

    实际上：当输入是 (batch, d_model) 时，我们把它当作
    序列长度=1的Transformer处理，主要用其FFN部分做特征变换。
    如果要做跨基因图的attention，请使用 MultiViewGCN。
    """
    def __init__(self, d_model: int, n_heads: int = 8,
                 ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, d_model) 或 (batch, seq, d_model)"""
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)   # (batch, 1, d_model)
            squeeze = True

        # Self-attention（Pre-Norm）
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out

        # FFN（Pre-Norm）
        x = x + self.ffn(self.norm2(x))

        if squeeze:
            x = x.squeeze(1)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# 主模型
# ══════════════════════════════════════════════════════════════════════════════

class PlantHGNN(nn.Module):
    """
    植物基因组预测异构图神经网络（主模型）

    Args:
        n_snps:              PCS后的SNP数量（输入特征维度）
        d_model:             隐层维度（默认128）
        n_transformer_layers: Transformer层数（默认6）
        n_attnres_blocks:    AttnRes block数（默认8）
        n_traits:            预测性状数（默认1）
        n_gcn_genes:         GCN节点数（基因数）
        n_views:             多视图数量（默认3: PPI+GO+Pathway）
        use_gcn:             是否启用MultiViewGCN（消融开关）
        use_attnres:         是否启用AttnRes（消融开关）
        dropout:             Dropout率
    """

    def __init__(
        self,
        n_snps: int,
        d_model: int = 128,
        n_transformer_layers: int = 6,
        n_attnres_blocks: int = 8,
        n_traits: int = 1,
        n_gcn_genes: int = 0,
        n_views: int = 3,
        use_gcn: bool = True,
        use_attnres: bool = True,
        n_heads: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.use_gcn     = use_gcn and (n_gcn_genes > 0)
        self.use_attnres = use_attnres
        self.d_model     = d_model
        self.n_attnres_blocks = n_attnres_blocks
        self.n_transformer_layers = n_transformer_layers

        # ── 1. SNP编码器 ─────────────────────────────────────────────────────
        self.snp_encoder = SNPEncoder(n_snps, d_model, dropout)

        # ── 2. (可选) MultiViewGCN ────────────────────────────────────────────
        if self.use_gcn:
            # GCN编码器：输入是基因级SNP特征（每基因1维均值）
            self.gcn_encoder = MultiViewGCNEncoder(
                in_dim=1, hidden_dim=d_model, out_dim=d_model,
                n_views=n_views, dropout=dropout
            )
            # GCN输出聚合到样本级别（全局平均池化后投影）
            self.gcn_pool = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
            )
            # 融合SNP特征和GCN特征
            self.fusion = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
            )

        # ── 3. Transformer + AttnRes ──────────────────────────────────────────
        # 将n_transformer_layers平均分配到n_attnres_blocks个block
        layers_per_block = max(1, n_transformer_layers // n_attnres_blocks)
        self.transformer_blocks = nn.ModuleList()
        total = 0
        for b in range(n_attnres_blocks):
            n_layers = layers_per_block
            if b == n_attnres_blocks - 1:
                n_layers = max(1, n_transformer_layers - total)
            block_layers = nn.ModuleList([
                TransformerBlock(d_model, n_heads=n_heads,
                                  ffn_mult=4, dropout=dropout)
                for _ in range(n_layers)
            ])
            self.transformer_blocks.append(block_layers)
            total += n_layers

        if self.use_attnres:
            self.attn_res = BlockAttnRes(d_model, n_attnres_blocks, dropout)
        else:
            self.attn_res = None

        # ── 4. 回归头 ──────────────────────────────────────────────────────────
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_traits),
        )

    def forward(
        self,
        snp: torch.Tensor,
        gene_feat: torch.Tensor = None,
        adj_list: list = None,
    ) -> torch.Tensor:
        """
        Args:
            snp:       (batch, n_snps) — 标准化SNP矩阵
            gene_feat: (batch, n_genes) — 基因级SNP均值特征（GCN用）
            adj_list:  list of (n_genes, n_genes) or None × n_views

        Returns:
            pred: (batch, n_traits) 或 (batch,) 当n_traits=1
        """
        # ── 1. SNP编码 ────────────────────────────────────────────────────────
        h = self.snp_encoder(snp)   # (batch, d_model)

        # ── 2. (可选) GCN融合 ─────────────────────────────────────────────────
        if self.use_gcn and gene_feat is not None and adj_list is not None:
            # gene_feat: (batch, n_genes) → (batch, n_genes, 1)
            x_gene = gene_feat.unsqueeze(-1)
            z_gcn, _ = self.gcn_encoder(x_gene, adj_list)  # (batch, n_genes, d_model)
            z_gcn_pool = z_gcn.mean(dim=1)                  # (batch, d_model)
            z_gcn_pool = self.gcn_pool(z_gcn_pool)          # (batch, d_model)
            h = self.fusion(torch.cat([h, z_gcn_pool], dim=-1))  # (batch, d_model)

        # ── 3. Transformer + AttnRes ──────────────────────────────────────────
        if self.use_attnres:
            self.attn_res.reset()

        for block_idx, block_layers in enumerate(self.transformer_blocks):
            # block内标准残差
            for layer in block_layers:
                h = layer(h)

            if self.use_attnres:
                self.attn_res.register_block_output(h)
                # 从第二个block开始，用AttnRes替代标准残差（Kimi AttnRes原始设计）
                if block_idx > 0:
                    attn_residual = self.attn_res(block_idx)
                    if attn_residual is not None:
                        h = h + attn_residual   # 全量AttnRes残差

        # ── 4. 预测 ───────────────────────────────────────────────────────────
        h = self.final_norm(h)
        out = self.head(h)   # (batch, n_traits)

        if out.shape[-1] == 1:
            out = out.squeeze(-1)  # (batch,)
        return out

    def get_network_attention_weights(self) -> torch.Tensor:
        """
        返回多视图GCN的视图注意力权重（用于可解释性）。
        """
        if self.use_gcn:
            return self.gcn_encoder.get_view_attention_weights()
        return None

    def get_depth_attention_weights(self) -> torch.Tensor:
        """
        返回AttnRes的深度方向注意力权重（用于可解释性）。
        """
        if self.use_attnres and self.attn_res is not None:
            return self.attn_res.get_block_attention_weights()
        return None

"""
MultiViewGCNEncoder: 多视图GCN编码器

对 PPI / GO / Pathway 三个网络分别运行2层GCN，
然后通过可学习注意力权重融合三个视图的基因嵌入。

输入: gene-level特征矩阵 x: (batch, n_genes, d_in)
输出: fused embedding z: (batch, n_genes, d_out)
      attention weights α: (3,) — 各视图的权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """单层GCN: h = activation(A_norm @ x @ W + b)"""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x:   (batch, n_genes, in_dim)
        adj: (n_genes, n_genes) 归一化邻接矩阵（D^{-1/2} A D^{-1/2}）
        """
        x = self.linear(x)  # (batch, n_genes, out_dim)
        # 批量矩阵乘法: adj @ x
        x = torch.matmul(adj.unsqueeze(0), x)  # (batch, n_genes, out_dim)
        return x


class SingleViewGCN(nn.Module):
    """
    2层GCN，处理单个网络视图。
    中间加BatchNorm1d和Dropout提升稳定性。
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 dropout: float = 0.2):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, out_dim)
        self.bn1  = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x:   (batch, n_genes, in_dim)
        adj: (n_genes, n_genes)
        """
        batch, n_genes, _ = x.shape

        # Layer 1
        h = self.gcn1(x, adj)   # (batch, n_genes, hidden_dim)
        # BN需要 (N, C) 格式，reshape
        h = h.reshape(-1, h.size(-1))
        h = self.bn1(h)
        h = h.reshape(batch, n_genes, -1)
        h = F.relu(h)
        h = self.dropout(h)

        # Layer 2
        h = self.gcn2(h, adj)   # (batch, n_genes, out_dim)
        return h


class MultiViewGCNEncoder(nn.Module):
    """
    三视图GCN编码器：PPI + GO + Pathway

    通过可学习注意力权重融合三个视图的基因嵌入。
    如果某个网络未提供（adj=None），则跳过该视图。

    Args:
        in_dim:     输入特征维度（每个基因的初始特征维度）
        hidden_dim: GCN隐层维度
        out_dim:    输出嵌入维度
        n_views:    视图数量（默认3: PPI, GO, Pathway）
        dropout:    Dropout率
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 n_views: int = 3, dropout: float = 0.2):
        super().__init__()
        self.n_views = n_views
        self.out_dim = out_dim

        # 每个视图独立的GCN
        self.gcns = nn.ModuleList([
            SingleViewGCN(in_dim, hidden_dim, out_dim, dropout)
            for _ in range(n_views)
        ])

        # 可学习注意力向量 a: (out_dim,) 用于视图间注意力计算
        self.view_attn = nn.Linear(out_dim, 1, bias=False)

        self._last_attn_weights = None   # 缓存用于可解释性

    def forward(self, x: torch.Tensor,
                adj_list: list) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:        (batch, n_genes, in_dim)
            adj_list: list of adj tensors (n_genes, n_genes) or None
                      长度为 n_views，None 表示该视图不可用

        Returns:
            z_fused:  (batch, n_genes, out_dim)
            attn_w:   (n_views,) 各视图的平均注意力权重
        """
        view_outputs = []
        valid_views  = []

        for i, (gcn, adj) in enumerate(zip(self.gcns, adj_list)):
            if adj is None:
                continue
            z_i = gcn(x, adj)          # (batch, n_genes, out_dim)
            view_outputs.append(z_i)
            valid_views.append(i)

        if not view_outputs:
            # Fallback: 无网络时直接用线性层
            return self.gcns[0].gcn1.linear(x), torch.ones(self.n_views) / self.n_views

        # 计算每个视图的注意力得分
        # attn_score[i] = mean over (batch, n_genes) of tanh(z_i) @ a
        stacked = torch.stack(view_outputs, dim=0)    # (n_valid, batch, n_genes, d)
        scores  = self.view_attn(torch.tanh(stacked)) # (n_valid, batch, n_genes, 1)
        scores  = scores.mean(dim=(1, 2))              # (n_valid, 1)
        attn_w  = torch.softmax(scores.squeeze(-1), dim=0)  # (n_valid,)

        # 加权求和
        z_fused = (stacked * attn_w[:, None, None, None]).sum(dim=0)  # (batch, n_genes, d)

        # 返回完整 n_views 长度的权重（未使用的视图权重为0）
        full_attn = torch.zeros(self.n_views, device=x.device)
        for rank, view_idx in enumerate(valid_views):
            full_attn[view_idx] = attn_w[rank]

        self._last_attn_weights = full_attn.detach()
        return z_fused, full_attn

    def get_view_attention_weights(self) -> torch.Tensor:
        """返回最近一次前向传播的视图注意力权重（用于可解释性）。"""
        return self._last_attn_weights

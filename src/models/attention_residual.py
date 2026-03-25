"""
Block Attention Residuals (AttnRes)
参考: Kimi arxiv 2603.15031 (MoonshotAI/Attention-Residuals)

核心思想：
  标准残差: h^(l) = h^(l-1) + F(h^(l-1))
  AttnRes:  h^(l) = Σ α_{i→l} · v_i  (跨所有之前block的注意力加权聚合)

  其中:
    - 将L层分为N个block（默认N=8）
    - block内部仍用标准残差
    - query向量 q_l 是可训练参数（不依赖当前输入）
    - key/value 来自各block的输出隐状态

这使得模型可以自适应地"决定"最终表示应主要来自哪一层，
提供可解释的深度方向注意力权重。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BlockAttnRes(nn.Module):
    """
    Block Attention Residuals 模块。

    将Transformer层划分为 n_blocks 个block，每个block结束时将
    其输出注册为一个"记忆"。最终输出通过对所有block输出做
    注意力加权聚合得到。

    Args:
        d_model: 隐层维度
        n_blocks: block数量（默认8，遵循Kimi论文推荐）
        n_layers_per_block: 每个block包含的Transformer层数
        dropout: attention dropout
    """

    def __init__(self, d_model: int, n_blocks: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_blocks = n_blocks

        # 每个block的可训练 query 向量（不依赖输入）
        self.queries = nn.Parameter(torch.randn(n_blocks, d_model) * 0.02)

        # Key/Value 投影（将block输出投影到注意力空间）
        self.key_proj   = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)

        self.scale   = math.sqrt(d_model)
        self.dropout = nn.Dropout(dropout)

        self._block_outputs = []   # 存储各block输出（前向传播时填充）

    def reset(self):
        """每次前向传播前重置block输出缓存。"""
        self._block_outputs = []

    def register_block_output(self, h: torch.Tensor):
        """
        在每个block结束时调用，注册该block的输出。
        h: (batch, seq_len, d_model) 或 (batch, d_model)
        """
        self._block_outputs.append(h.detach() if not self.training else h)

    def forward(self, current_block_idx: int) -> torch.Tensor:
        """
        为当前block计算注意力加权聚合的"增强残差"。

        Args:
            current_block_idx: 当前block的索引（0-based）

        Returns:
            attn_residual: 形状同 block_outputs[0]，作为额外的残差加入
        """
        if not self._block_outputs:
            return None

        # 收集已有的block输出: list of (batch, ..., d_model)
        # 统一处理: 不论是1D还是2D序列
        outputs = self._block_outputs  # 已注册的所有block

        # Stack: (n_registered, batch, d_model) 或 (n_registered, batch, seq, d_model)
        # 为统一处理，将h展平到最后一维
        h0 = outputs[0]
        seq_shape = h0.shape[:-1]   # (batch,) 或 (batch, seq)
        d = h0.shape[-1]

        # (n_prev, batch, ..., d) → (n_prev, -1, d)
        stacked = torch.stack(outputs, dim=0)   # (n_prev, batch, ..., d)
        n_prev   = stacked.shape[0]
        flat_shape = stacked.shape[1:-1]         # (batch, ...)
        flat_n = 1
        for s in flat_shape:
            flat_n *= s
        stacked_2d = stacked.reshape(n_prev, flat_n, d)   # (n_prev, N, d)

        # Keys/Values: (n_prev, N, d)
        keys   = self.key_proj(stacked_2d)    # (n_prev, N, d)
        values = self.value_proj(stacked_2d)  # (n_prev, N, d)

        # Query: (d,) → (1, 1, d) for broadcasting
        q = self.queries[current_block_idx]   # (d,)
        q = q.unsqueeze(0).unsqueeze(0)       # (1, 1, d)

        # Attention scores: (n_prev, N, 1)
        scores = (keys * q).sum(dim=-1, keepdim=True) / self.scale  # (n_prev, N, 1)
        # Softmax over n_prev dimension
        scores = scores.permute(1, 0, 2)       # (N, n_prev, 1)
        attn   = F.softmax(scores, dim=1)       # (N, n_prev, 1)
        attn   = self.dropout(attn)

        # Weighted sum of values: (N, d)
        values_perm = values.permute(1, 0, 2)   # (N, n_prev, d)
        aggregated  = (attn * values_perm).sum(dim=1)  # (N, d)

        # Reshape back to original
        aggregated = aggregated.reshape(*flat_shape, d)  # (batch, ..., d)

        return aggregated

    def get_block_attention_weights(self) -> torch.Tensor:
        """
        返回各block的注意力权重（用于可解释性分析）。

        Returns:
            weights: (n_blocks, n_blocks) 矩阵，
                     weights[i, j] = block i 对 block j输出的注意力权重
        """
        if not self._block_outputs:
            return None

        # 创建虚拟前向，提取权重
        outputs = self._block_outputs
        n_reg = len(outputs)
        h0 = outputs[0]
        d = h0.shape[-1]

        stacked = torch.stack(outputs, dim=0)
        flat_n = stacked[0].numel() // d
        stacked_2d = stacked.reshape(n_reg, flat_n, d)
        keys = self.key_proj(stacked_2d)

        all_weights = []
        for i in range(min(self.n_blocks, n_reg)):
            q = self.queries[i].unsqueeze(0).unsqueeze(0)
            scores = (keys * q).sum(dim=-1).mean(dim=1) / self.scale  # (n_reg,)
            w = F.softmax(scores, dim=0)
            all_weights.append(w)

        return torch.stack(all_weights, dim=0)   # (n_blocks, n_reg)

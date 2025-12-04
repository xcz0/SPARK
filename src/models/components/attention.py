"""多头差异化注意力机制 (Multi-Head Differential Attention)。

根据模型架构文档（第3节）：
将 H 个注意力头划分为三组，每组应用不同的 Attention Mask 和 Bias 策略：
- 组 A: 卡片记忆回溯头 (Card-Specific Heads)
- 组 B: 知识关联头 (Concept/Deck Heads)
- 组 C: 全局上下文头 (Global Context Heads)
"""

import torch
from einops import rearrange, repeat
from torch.nn import Module, Parameter, MultiheadAttention
from torch.nn.functional import softplus
from torch import Tensor


class TimeDecayBias(Module):
    """时间衰减偏置模块。

    公式: Bias_decay(i, j) = w_card * exp(-λ * |T_i - T_j|)
    """

    def __init__(self, num_heads: int):
        """初始化时间衰减偏置。

        Args:
            num_heads: 使用时间衰减的头数
        """
        super().__init__()
        # 可学习的权重 w_card
        self.weight = Parameter(torch.ones(num_heads))
        # 可学习的衰减系数 λ（通过 softplus 约束为正）
        self.decay_raw = Parameter(torch.zeros(num_heads))

    def _get_decay(self) -> Tensor:
        """获取衰减系数（保证 > 0）。"""
        return softplus(self.decay_raw)

    def forward(self, time_diff: Tensor) -> Tensor:
        """计算时间衰减偏置。

        Args:
            time_diff: (batch, seq_len, seq_len) 时间差矩阵

        Returns:
            (batch, num_heads, seq_len, seq_len) 时间衰减偏置
        """
        # 扩展 time_diff 到多头维度
        time_diff = rearrange(
            time_diff, "batch src_pos tgt_pos -> batch 1 src_pos tgt_pos"
        )

        # 获取参数值并扩展到广播形状
        decay = rearrange(self._get_decay(), "heads -> 1 heads 1 1")
        weight = rearrange(self.weight, "heads -> 1 heads 1 1")

        # 计算衰减偏置
        return weight * torch.exp(-decay * time_diff)


class DifferentialMultiHeadAttention(Module):
    """多头差异化注意力。

    将注意力头分为三组，分别应用不同的掩码策略：
    - Card Heads: 仅关注同一卡片的历史 + 时间衰减
    - Deck Heads: 关注同一卡组的其他卡片
    - Global Heads: 全局因果注意力
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        card_head_ratio: float = 0.5,
        deck_head_ratio: float = 0.25,
    ):
        """初始化多头差异化注意力。

        Args:
            d_model: 模型维度
            n_heads: 总注意力头数
            dropout: Dropout 概率
            card_head_ratio: 卡片头占比
            deck_head_ratio: 卡组头占比
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.d_model = d_model
        self.n_heads = n_heads

        # 计算各组头的数量
        self.n_card_heads = max(1, int(n_heads * card_head_ratio))
        self.n_deck_heads = max(1, int(n_heads * deck_head_ratio))
        self.n_global_heads = n_heads - self.n_card_heads - self.n_deck_heads
        assert self.n_global_heads >= 1, "全局头数量必须至少为 1"

        # 使用 PyTorch 的 MultiheadAttention
        self.mha = MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 时间衰减偏置（仅用于 Card Heads）
        self.time_decay = TimeDecayBias(self.n_card_heads)

    def forward(
        self,
        x: Tensor,
        causal_mask: Tensor,
        card_mask: Tensor,
        deck_mask: Tensor,
        time_diff: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """前向传播。

        Args:
            x: (batch, seq_len, d_model) 输入
            causal_mask: (seq_len, seq_len) 因果掩码
            card_mask: (batch, seq_len, seq_len) 卡片掩码
            deck_mask: (batch, seq_len, seq_len) 卡组掩码
            time_diff: (batch, seq_len, seq_len) 时间差矩阵
            padding_mask: (batch, seq_len) 填充掩码

        Returns:
            (batch, seq_len, d_model) 输出
        """
        batch_size, seq_len, _ = x.shape

        # 扩展因果掩码到批次维度
        causal_mask = repeat(
            causal_mask, "src_pos tgt_pos -> batch src_pos tgt_pos", batch=batch_size
        )

        # 处理填充掩码
        if padding_mask is not None:
            padding_mask_2d = padding_mask.unsqueeze(1) & padding_mask.unsqueeze(2)
            causal_mask = causal_mask & padding_mask_2d
            card_mask = card_mask & padding_mask_2d
            deck_mask = deck_mask & padding_mask_2d

        # 转换掩码为加性掩码 (0 或 -inf)
        def to_additive(mask):
            return torch.zeros_like(mask, dtype=x.dtype).masked_fill(
                ~mask, float("-inf")
            )

        card_attn_mask = to_additive(card_mask)
        deck_attn_mask = to_additive(deck_mask)
        causal_attn_mask = to_additive(causal_mask)

        # === Card Heads Mask ===
        # 时间衰减偏置: (batch, n_card_heads, seq_len, seq_len)
        time_bias = self.time_decay(time_diff)
        # 广播 card_attn_mask: (batch, 1, seq_len, seq_len)
        card_attn_mask = card_attn_mask.unsqueeze(1) + time_bias

        # === Deck Heads Mask ===
        # (batch, n_deck_heads, seq_len, seq_len)
        deck_attn_mask = deck_attn_mask.unsqueeze(1).expand(
            -1, self.n_deck_heads, -1, -1
        )

        # === Global Heads Mask ===
        # (batch, n_global_heads, seq_len, seq_len)
        causal_attn_mask = causal_attn_mask.unsqueeze(1).expand(
            -1, self.n_global_heads, -1, -1
        )

        # 合并所有头的掩码
        # (batch, n_heads, seq_len, seq_len)
        combined_mask = torch.cat(
            [card_attn_mask, deck_attn_mask, causal_attn_mask], dim=1
        )

        # 重塑为 (batch * n_heads, seq_len, seq_len) 以供 MultiheadAttention 使用
        combined_mask = rearrange(combined_mask, "b h s t -> (b h) s t")

        # MultiheadAttention 前向传播
        attn_output, _ = self.mha(
            query=x, key=x, value=x, attn_mask=combined_mask, need_weights=False
        )

        return attn_output

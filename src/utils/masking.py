"""
用于注意力机制的掩码工具函数。所有函数返回 PyTorch 张量。

包含基础掩码函数和复合掩码函数：
- 基础掩码：padding、causal、same/different element
- 复合掩码：card mask、deck mask（用于多头注意力）
- 时间差矩阵：用于时间衰减偏置
"""

from functools import reduce
from typing import Optional

import torch
from torch import Tensor

# ============================================================================
# 基础掩码函数
# ============================================================================


def create_padding_mask(seq_lens: Tensor, max_len: Optional[int] = None) -> Tensor:
    """创建填充掩码，标记有效位置 (True为有效)。

    Args:
        seq_lens: (batch,) 序列长度
        max_len: 最大长度，若为 None 则自动推导

    Returns:
        (batch, max_len) BoolTensor
    """
    if max_len is None:
        # 使用 max() 会触发同步，但在 CPU Collator 中通常可接受
        max_len = int(seq_lens.max().item())

    # 利用广播：(1, max_len) < (batch, 1) -> (batch, max_len)
    positions = torch.arange(max_len, device=seq_lens.device).unsqueeze(0)
    return positions < seq_lens.unsqueeze(1)


def create_causal_mask(seq_len: int, device: torch.device) -> Tensor:
    """创建因果掩码。

    Returns:
        (seq_len, seq_len) BoolTensor, 下三角为 True
    """
    # 使用 tril 直接生成下三角布尔矩阵，避免中间分配
    return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))


def create_same_element_mask(ids: Tensor, special_id: Optional[int] = None) -> Tensor:
    """创建同元素掩码。

    Args:
        ids: (batch, seq_len)
        special_id: 如果提供，该 ID 之间的位置将被视为 False (不关注)

    Returns:
        (batch, seq_len, seq_len) BoolTensor
    """
    # 使用 None 索引进行广播：(batch, seq_len, 1) == (batch, 1, seq_len)
    same_mask = ids[:, :, None] == ids[:, None, :]

    if special_id is None:
        return same_mask

    # 只有当两个位置都不是 special_id 时才保留
    valid_ids = ids != special_id
    valid_pairs = valid_ids[:, :, None] & valid_ids[:, None, :]
    return same_mask & valid_pairs


def create_different_element_mask(ids: Tensor) -> Tensor:
    """创建不同元素掩码。"""
    return ids[:, :, None] != ids[:, None, :]


def combine_masks(*masks: Tensor) -> Tensor:
    """组合多个掩码（逻辑与）。"""
    if not masks:
        raise ValueError("Must provide at least one mask.")
    return reduce(torch.logical_and, masks)


def create_padding_mask_2d(padding_mask: Tensor) -> Tensor:
    """将 1D (batch, seq) 填充掩码转为 2D (batch, seq, seq)。"""
    # (batch, seq, 1) & (batch, 1, seq)
    return padding_mask[:, :, None] & padding_mask[:, None, :]


def apply_padding_to_attention_mask(
    attention_mask: Tensor, padding_mask: Tensor
) -> Tensor:
    """将填充掩码应用到注意力掩码。"""
    padding_2d = create_padding_mask_2d(padding_mask)

    # 如果 attention_mask 是 (L, L)，它会自动广播到 (B, L, L)
    return attention_mask & padding_2d


# ============================================================================
# 复合掩码函数
# ============================================================================


def create_card_mask(card_ids: Tensor) -> Tensor:
    """创建卡片掩码 (Same Card & Causal)。"""
    # 这里的 combine 能够处理不同维度的广播
    # (B, L, L) & (L, L) -> (B, L, L)
    return combine_masks(
        create_same_element_mask(card_ids),
        create_causal_mask(card_ids.size(1), device=card_ids.device),
    )


def create_deck_mask(deck_ids: Tensor, card_ids: Tensor) -> Tensor:
    """创建卡组掩码 (Same Deck & Different Card & Causal)。"""
    return combine_masks(
        create_same_element_mask(ids=deck_ids, special_id=0),  #  0 是特殊卡组 ID
        create_different_element_mask(card_ids),
        create_causal_mask(deck_ids.size(1), device=deck_ids.device),
    )


def create_time_diff_matrix(time_stamps: Tensor) -> Tensor:
    """创建时间差矩阵 (绝对值)。"""
    # (batch, seq, 1) - (batch, 1, seq) -> (batch, seq, seq)
    return (time_stamps[:, :, None] - time_stamps[:, None, :]).abs()

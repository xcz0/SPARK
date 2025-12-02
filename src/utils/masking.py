"""
用于注意力机制的掩码工具函数。所有函数返回 PyTorch 张量。

包含基础掩码函数和复合掩码函数：
- 基础掩码：padding、causal、same/different element
- 复合掩码：card mask、deck mask（用于多头注意力）
- 时间差矩阵：用于时间衰减偏置
"""

from functools import reduce

import torch
from einops import rearrange
from torch import Tensor


# ============================================================================
# 基础掩码函数
# ============================================================================


def create_padding_mask(seq_lens: Tensor, max_len: int | None) -> Tensor:
    """创建填充掩码，标记有效位置。

    Args:
        seq_lens: (batch,) 每个序列的实际长度
        max_len: 最大序列长度，默认使用 seq_lens 中的最大值

    Returns:
        (batch, max_len) 布尔张量，True 表示有效位置
    """
    max_len = max_len or int(seq_lens.max().item())
    positions = torch.arange(max_len, device=seq_lens.device)
    return rearrange(positions, "seq -> 1 seq") < rearrange(
        seq_lens, "batch -> batch 1"
    )


def create_causal_mask(seq_len: int, device: torch.device) -> Tensor:
    """创建因果掩码，防止注意到未来位置。

    Args:
        seq_len: 序列长度
        device: 目标设备

    Returns:
        (seq_len, seq_len) 布尔下三角矩阵
    """
    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()


def create_same_element_mask(ids: Tensor, special_id: int | None = None) -> Tensor:
    """创建同元素掩码，标记 ID 相同的位置对。

    适用于标记同一卡片、同一用户等场景。

    Args:
        ids: (batch, seq_len) 元素 ID 张量
        special_id: 特殊 ID，该 ID 对应的位置掩码标记为 False

    Returns:
        (batch, seq_len, seq_len) 布尔张量
    """
    same_mask = rearrange(ids, "batch seq_i -> batch 1 seq_i") == rearrange(
        ids, "batch seq_j -> batch seq_j 1"
    )

    if special_id is None:
        return same_mask

    # 特殊 ID 对应的位置标记为 False
    special_mask = rearrange(ids != special_id, "batch seq -> batch 1 seq") & rearrange(
        ids != special_id, "batch seq -> batch seq 1"
    )
    return same_mask & special_mask


def create_different_element_mask(ids: Tensor) -> Tensor:
    """创建不同元素掩码，标记 ID 不同的位置对。

    Args:
        ids: (batch, seq_len) 元素 ID 张量

    Returns:
        (batch, seq_len, seq_len) 布尔张量
    """
    return rearrange(ids, "batch seq_i -> batch 1 seq_i") != rearrange(
        ids, "batch seq_j -> batch seq_j 1"
    )


def combine_masks(*masks: Tensor) -> Tensor:
    """组合多个布尔掩码（逻辑与）。

    自动处理维度广播。

    Args:
        *masks: 多个布尔掩码张量

    Returns:
        组合后的掩码
    """
    return reduce(torch.Tensor.__and__, masks)


def create_padding_mask_2d(padding_mask: Tensor) -> Tensor:
    """将 1D 填充掩码转换为 2D 注意力掩码。

    Args:
        padding_mask: (batch, seq_len) 填充掩码

    Returns:
        (batch, seq_len, seq_len) 2D 填充掩码
    """
    return rearrange(padding_mask, "batch seq_i -> batch 1 seq_i") & rearrange(
        padding_mask, "batch seq_j -> batch seq_j 1"
    )


def apply_padding_to_attention_mask(
    attention_mask: Tensor, padding_mask: Tensor
) -> Tensor:
    """将填充掩码应用到注意力掩码上。

    Args:
        attention_mask: (batch, seq_len, seq_len) 或 (seq_len, seq_len) 注意力掩码
        padding_mask: (batch, seq_len) 填充掩码

    Returns:
        (batch, seq_len, seq_len) 应用填充后的掩码
    """
    padding_2d = create_padding_mask_2d(padding_mask)

    if attention_mask.dim() == 2:
        attention_mask = rearrange(attention_mask, "query key -> 1 query key")

    return attention_mask & padding_2d


# ============================================================================
# 复合掩码函数（用于多头注意力机制）
# ============================================================================


def create_card_mask(card_ids: Tensor) -> Tensor:
    """创建卡片掩码，用于卡片记忆回溯头。

    仅允许关注同一张卡片的历史记录。

    Args:
        card_ids: (batch, seq_len) 卡片 ID 张量

    Returns:
        (batch, seq_len, seq_len) 布尔掩码
    """
    return combine_masks(
        create_same_element_mask(card_ids),
        create_causal_mask(card_ids.size(1), device=card_ids.device),
    )


def create_deck_mask(deck_ids: Tensor, card_ids: Tensor) -> Tensor:
    """创建卡组掩码，用于知识关联头。

    允许关注同一卡组的其他卡片（排除同一张卡片）。

    Args:
        deck_ids: (batch, seq_len) 卡组 ID 张量
        card_ids: (batch, seq_len) 卡片 ID 张量

    Returns:
        (batch, seq_len, seq_len) 布尔掩码
    """
    return combine_masks(
        create_same_element_mask(deck_ids),
        create_different_element_mask(card_ids),
        create_causal_mask(deck_ids.size(1), device=deck_ids.device),
    )


def create_time_diff_matrix(time_stamps: Tensor) -> Tensor:
    """创建时间差矩阵，用于计算时间衰减偏置。

    Args:
        time_stamps: (batch, seq_len) 时间戳张量

    Returns:
        (batch, seq_len, seq_len) 时间差绝对值矩阵
    """
    return torch.abs(
        rearrange(time_stamps, "batch query -> batch query 1")
        - rearrange(time_stamps, "batch key -> batch 1 key")
    )

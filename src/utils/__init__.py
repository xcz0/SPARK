"""工具模块。

提供掩码生成和评估指标等功能。

掩码函数分类：
- 基础掩码：padding、causal、same/different element
- 复合掩码：card mask、deck mask（用于多头注意力）
- 时间差矩阵：用于时间衰减偏置
"""

from torchmetrics import MeanSquaredError

from .masking import (
    apply_padding_to_attention_mask,
    combine_masks,
    create_card_mask,
    create_causal_mask,
    create_deck_mask,
    create_different_element_mask,
    create_padding_mask,
    create_padding_mask_2d,
    create_same_element_mask,
    create_time_diff_matrix,
)
from .metrics import OrdinalAccuracy, ordinal_accuracy, rmse

# 向后兼容别名：使用 torchmetrics 内置的 RMSE
MaskedRMSE = lambda **kwargs: MeanSquaredError(squared=False, **kwargs)

__all__ = [
    # 基础掩码函数
    "create_causal_mask",
    "create_padding_mask",
    "create_padding_mask_2d",
    "create_same_element_mask",
    "create_different_element_mask",
    "combine_masks",
    "apply_padding_to_attention_mask",
    # 复合掩码函数
    "create_card_mask",
    "create_deck_mask",
    "create_time_diff_matrix",
    # 指标类
    "OrdinalAccuracy",
    "MaskedRMSE",  # 向后兼容别名
    # 函数式指标
    "rmse",
    "ordinal_accuracy",
]

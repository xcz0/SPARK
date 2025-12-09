"""数据处理模块。

提供复习序列数据的加载、处理和批处理功能。

模块结构：
- config: 特征配置和 Collator 配置
- dataset: 复习序列数据集（预加载和流式）
- collator: 动态填充和掩码生成
- loader: 数据加载工具函数
- datamodule: PyTorch Lightning 数据模块
"""

from .collator import ReviewCollator
from .config import (
    CollatorConfig,
    FeatureConfig,
)
from .datamodule import ReviewDataModule
from .dataset import ReviewSequenceDataset, StreamingReviewDataset
from .loader import load_all_users_data, load_user_data
from ..utils.masking import (
    create_card_mask,
    create_causal_mask,
    create_deck_mask,
    create_padding_mask,
    create_time_diff_matrix,
)

__all__ = [
    # 配置
    "FeatureConfig",
    "CollatorConfig",
    # 数据集
    "ReviewSequenceDataset",
    "StreamingReviewDataset",
    # Collator
    "ReviewCollator",
    # 掩码函数（从 utils.masking 重新导出）
    "create_causal_mask",
    "create_card_mask",
    "create_deck_mask",
    "create_time_diff_matrix",
    "create_padding_mask",
    # DataModule
    "ReviewDataModule",
    # 工具函数
    "load_user_data",
    "load_all_users_data",
]

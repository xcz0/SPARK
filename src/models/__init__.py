"""Model package exports.

SPARK: Spaced Repetition with Attention-based Recall Knowledge model.
"""

from .architecture import ModelConfig, SPARKModel
from .losses import CombinedLoss, CORALLoss, DurationLoss
from .module import SPARKModule


__all__ = [
    # 核心模型
    "SPARKModel",
    "ModelConfig",
    # 损失函数
    "CORALLoss",
    "DurationLoss",
    "CombinedLoss",
    # Lightning 模块
    "SPARKModule",
]

"""损失函数实现。

- 评分预测：CORAL Loss = Σ BCE(output_k, I(y_true > k))
- 耗时预测：MSE Loss
"""

import torch
from torch.nn.functional import binary_cross_entropy, mse_loss
from torch import Tensor
from torch.nn import Module


def _reduce_loss(loss: Tensor, mask: Tensor | None, reduction: str) -> Tensor:
    """通用的损失聚合函数。

    Args:
        loss: 未聚合的损失张量
        mask: 有效位置掩码
        reduction: 聚合方式 ('mean', 'sum', 'none')

    Returns:
        聚合后的损失
    """
    if mask is not None:
        loss = loss * mask.float()
        if reduction == "mean":
            return loss.sum() / mask.float().sum().clamp(min=1.0)
        if reduction == "sum":
            return loss.sum()
        return loss

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


class CORALLoss(Module):
    """CORAL (Consistent Rank Logits) 有序回归损失。

    将有序分类问题转化为多个二分类问题，损失函数为：
    Loss = Σ_{k=1}^{K-1} BCE(P(y > k), I(y_true > k))
    """

    def __init__(self, reduction: str = "mean"):
        """初始化 CORAL 损失。

        Args:
            reduction: 损失聚合方式，'mean', 'sum' 或 'none'
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        rating_probs: Tensor,
        ordinal_targets: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """计算 CORAL 损失。

        Args:
            rating_probs: (batch, seq_len, num_thresholds) 模型输出的累积概率
            ordinal_targets: (batch, seq_len, num_thresholds) 有序目标
            mask: (batch, seq_len) 有效位置掩码

        Returns:
            损失值
        """
        loss = binary_cross_entropy(rating_probs, ordinal_targets, reduction="none")
        loss = loss.sum(dim=-1)  # (batch, seq_len)
        return _reduce_loss(loss, mask, self.reduction)


class DurationLoss(Module):
    """耗时预测 MSE 损失。"""

    def __init__(self, reduction: str = "mean"):
        """初始化耗时损失。

        Args:
            reduction: 损失聚合方式
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        duration_pred: Tensor,
        duration_target: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """计算 MSE 损失。

        Args:
            duration_pred: (batch, seq_len) 预测的 log(duration + 1)
            duration_target: (batch, seq_len) 目标 log(duration + 1)
            mask: (batch, seq_len) 有效位置掩码

        Returns:
            损失值
        """
        loss = mse_loss(duration_pred, duration_target, reduction="none")
        return _reduce_loss(loss, mask, self.reduction)


class CombinedLoss(Module):
    """组合损失：评分损失 + 耗时损失。"""

    def __init__(
        self,
        rating_weight: float = 1.0,
        duration_weight: float = 0.1,
        reduction: str = "mean",
    ):
        """初始化组合损失。

        Args:
            rating_weight: 评分损失权重
            duration_weight: 耗时损失权重
            reduction: 损失聚合方式
        """
        super().__init__()
        self.register_buffer("rating_weight", torch.tensor(rating_weight))
        self.register_buffer("duration_weight", torch.tensor(duration_weight))
        self.rating_loss = CORALLoss(reduction=reduction)
        self.duration_loss = DurationLoss(reduction=reduction)

    def forward(
        self,
        rating_probs: Tensor,
        ordinal_targets: Tensor,
        duration_pred: Tensor,
        duration_target: Tensor,
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """计算组合损失。

        Args:
            rating_probs: 评分累积概率
            ordinal_targets: 有序目标
            duration_pred: 耗时预测
            duration_target: 耗时目标
            mask: 有效位置掩码

        Returns:
            包含 'total', 'rating', 'duration' 的损失字典
        """
        rating_loss = self.rating_loss(rating_probs, ordinal_targets, mask)
        duration_loss = self.duration_loss(duration_pred, duration_target, mask)

        total_loss = (
            self.rating_weight * rating_loss + self.duration_weight * duration_loss
        )

        return {
            "total": total_loss,
            "rating": rating_loss,
            "duration": duration_loss,
        }

"""评估指标：使用 TorchMetrics 实现的有序分类准确率等。

提供两种使用方式：
1. TorchMetrics 类：用于训练循环中的状态累积
2. 函数式接口：用于简单的批次计算

注意：对于 RMSE，建议使用 torchmetrics.MeanSquaredError(squared=False)
"""

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import mean_squared_error


class OrdinalAccuracy(Metric):
    """有序分类准确率指标。

    计算 CORAL 输出的预测评分与真实评分的匹配准确率。
    预测评分 = 1 + Σ I(P(y > k) > 0.5)
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    correct: Tensor
    total: Tensor

    def __init__(self, **kwargs):
        """初始化有序准确率指标。"""
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """更新指标状态。

        Args:
            preds: (N,) 或 (N, num_thresholds) 预测值
                   如果是累积概率，自动转换为评分
            targets: (N,) 或 (N, num_thresholds) 真实评分表示
        """
        # 如果是累积概率，转换为评分
        if preds.dim() == 2:
            preds = 1 + (preds > 0.5).sum(dim=-1)

        if targets.dim() == 2:
            targets = 1 + targets.sum(dim=-1)

        targets = targets.to(device=preds.device, dtype=preds.dtype)

        self.correct += (preds == targets).sum()
        self.total += targets.numel()

    def compute(self) -> Tensor:
        """计算准确率。"""
        return self.correct.float() / self.total.clamp(min=1)


# 函数式接口
def rmse(preds: Tensor, targets: Tensor) -> Tensor:
    """计算 RMSE（函数式接口）。

    Args:
        preds: 预测张量
        targets: 目标张量

    Returns:
        RMSE 值
    """
    return torch.sqrt(mean_squared_error(preds, targets))


def ordinal_accuracy(preds: Tensor, targets: Tensor) -> Tensor:
    """计算有序分类准确率（函数式接口）。

    Args:
        preds: (N,) 或 (N, num_thresholds) 预测值
        targets: (N,) 或 (N, num_thresholds) 真实评分表示

    Returns:
        准确率
    """
    pred_scores = 1 + (preds > 0.5).sum(dim=-1) if preds.dim() == 2 else preds
    target_scores = 1 + targets.sum(dim=-1) if targets.dim() == 2 else targets
    target_scores = target_scores.to(device=pred_scores.device, dtype=pred_scores.dtype)

    return (pred_scores == target_scores).float().mean()

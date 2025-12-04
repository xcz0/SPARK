"""预测头：CORAL 有序回归和耗时预测。

根据模型架构文档（第4节-输出层设计）：
- Head 1: 评分预测 (Ordinal Regression) - 使用 CORAL 框架
- Head 2: 耗时预测 (Duration Regression) - MSE Loss
"""

import torch
from einops import rearrange
from torch.nn import Module, Linear, Parameter, Sequential, GELU, Dropout
from torch import Tensor


class CORALHead(Module):
    """CORAL (Consistent Rank Logits) 有序回归头。

    将 4 分制评分转化为 3 个二分类子任务：{P(y>1), P(y>2), P(y>3)}。
    共享同一个权重向量，但拥有 3 个独立的 bias，保证逻辑值的单调性。

    公式:
        logits = Linear(H_L) -> R^1
        output_k = σ(logits + b_k) for k in {1, 2, 3}
    """

    def __init__(self, d_model: int, num_classes: int = 4):
        """初始化 CORAL 头。

        Args:
            d_model: 输入特征维度
            num_classes: 评分类别数（默认 4 表示 1-4 分）
        """
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1  # 3 个阈值

        # 共享权重的线性层（输出维度为 1）
        self.shared_fc = Linear(d_model, 1)

        # 独立的 bias 参数（保证单调性需要初始化为递减）
        # b_1 > b_2 > b_3 对应 P(y>1) > P(y>2) > P(y>3)
        self.biases = Parameter(torch.linspace(1.0, -1.0, self.num_thresholds))

    def forward(self, x: Tensor) -> Tensor:
        """计算有序分类的累积概率。

        Args:
            x: (batch, seq_len, d_model) 输入特征

        Returns:
            (batch, seq_len, num_thresholds) 累积概率 P(y > k)
        """
        # 共享权重计算 logits
        logits = self.shared_fc(x)  # (batch, seq_len, 1)

        # 添加独立 bias 并计算 sigmoid
        # logits: (batch, seq_len, 1), biases: (num_thresholds,)
        # 广播为 (batch, seq_len, num_thresholds)
        cumulative_logits = logits + self.biases
        return torch.sigmoid(cumulative_logits)

    def predict_rating(self, x: Tensor) -> Tensor:
        """预测评分。

        解码公式: Pred = 1 + Σ I(output_k > 0.5)

        Args:
            x: (batch, seq_len, d_model) 输入特征

        Returns:
            (batch, seq_len) 预测评分 (1-4)
        """
        probs = self.forward(x)  # (batch, seq_len, num_thresholds)
        # 统计有多少个阈值被超过
        return 1 + (probs > 0.5).sum(dim=-1)

    def predict_expected_rating(self, x: Tensor) -> Tensor:
        """使用概率期望预测评分。

        Args:
            x: (batch, seq_len, d_model) 输入特征

        Returns:
            (batch, seq_len) 期望评分
        """
        probs = self.forward(x)  # (batch, seq_len, num_thresholds)
        # 期望 = 1 + sum(P(y > k))
        return 1.0 + probs.sum(dim=-1)

    def predict_correct(self, x: Tensor) -> Tensor:
        """预测回忆正确（评分大于1）的概率，即输出的第一个概率

        Args:
            x: (batch, seq_len, d_model) 输入特征

        Returns:
            (batch, seq_len) 预测回忆正确（评分大于1）
        """
        probs = self.forward(x)  # (batch, seq_len, num_thresholds)
        return probs[..., 0]  # P(y > 1)


class DurationHead(Module):
    """耗时预测头。

    预测 log(duration + 1)，使用 MSE Loss。
    """

    def __init__(
        self, d_model: int, hidden_dim: int | None = None, dropout: float = 0.1
    ):
        """初始化耗时预测头。

        Args:
            d_model: 输入特征维度
            hidden_dim: 隐藏层维度，默认为 d_model // 2
            dropout: Dropout 概率
        """
        super().__init__()
        hidden_dim = hidden_dim or d_model // 2

        self.mlp = Sequential(
            Linear(d_model, hidden_dim),
            GELU(),
            Dropout(dropout),
            Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """预测 log(duration + 1)。

        Args:
            x: (batch, seq_len, d_model) 输入特征

        Returns:
            (batch, seq_len) 预测的 log duration
        """
        output = self.mlp(x)  # (batch, seq_len, 1)
        return rearrange(output, "batch seq_len 1 -> batch seq_len")

"""Embedding 组件：Time2Vec 时间编码和类别特征嵌入。

- Time2Vec: 使用可学习的正弦波函数捕获周期性和线性时间模式
- 类别特征 Embedding: 对 state, rating, is_first_review 等进行嵌入
- 数值特征投影: 通过 MLP 将数值特征投影到指定维度
"""

import torch
from einops import rearrange
from torch.nn import (
    Module,
    Linear,
    Embedding,
    Parameter,
    Sequential,
    GELU,
    Dropout,
)
from torch import Tensor


class Time2Vec(Module):
    """Time2Vec 连续时间编码。

    使用可学习的正弦波函数捕获周期性和线性时间模式，替代 Transformer 的 Positional Encoding。

    公式:
        T2V(τ)[k] = ω_k * τ + φ_k           if k = 0 (线性项，捕捉长期趋势)
        T2V(τ)[k] = sin(ω_k * τ + φ_k)      if 1 ≤ k < K (周期项，捕捉遗忘/复习周期)
    """

    def __init__(self, embed_dim: int):
        """初始化 Time2Vec。

        Args:
            embed_dim: 时间编码维度 K
        """
        super().__init__()
        self.embed_dim = embed_dim

        # 可学习参数：频率和相位
        self.omega = Parameter(torch.randn(embed_dim))
        self.phi = Parameter(torch.zeros(embed_dim))

    def forward(self, timestamps: Tensor) -> Tensor:
        """计算时间编码。

        Args:
            timestamps: (batch, seq_len) 时间戳张量（单位：天）

        Returns:
            (batch, seq_len, embed_dim) 时间编码
        """
        # 扩展时间戳到嵌入维度
        t = rearrange(timestamps, "batch seq_len -> batch seq_len 1")

        # 计算 ω * τ + φ
        linear_term = self.omega * t + self.phi  # (batch, seq_len, embed_dim)

        # 第一维是线性项，其余是周期项
        periodic = torch.sin(linear_term[..., 1:])  # (batch, seq_len, embed_dim-1)
        linear = linear_term[..., :1]  # (batch, seq_len, 1)

        return torch.cat([linear, periodic], dim=-1)


class CategoricalEmbedding(Module):
    """类别特征嵌入层。

    将多个类别特征分别嵌入后拼接，再投影到目标维度。
    采用 Offset Trick 将所有特征合并到一个 Embedding 层中以提高效率。

    Warning:
        输入张量 `categorical_features` 的最后一维特征顺序必须与 `vocab_sizes` 字典的键顺序严格一致。
    """

    def __init__(
        self,
        vocab_sizes: dict[str, int],
        embed_dim: int,
        output_dim: int,
    ):
        """初始化类别嵌入层。

        Args:
            vocab_sizes: 各类别特征的词表大小。
                         注意：字典键的顺序必须与输入张量最后一维的特征顺序一致。
            embed_dim: 每个类别特征的嵌入维度
            output_dim: 最终输出维度
        """
        super().__init__()
        self.feature_names = list(vocab_sizes.keys())
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # 优化：使用单个 Embedding 层 + Offsets 处理所有特征
        # 这比使用 ModuleDict 和循环更高效，减少了 CUDA kernel launch

        # 1. 计算每个特征的 offset
        sizes = list(vocab_sizes.values())
        # offsets: [0, size_0, size_0+size_1, ...]
        offsets = torch.tensor([0] + sizes[:-1], dtype=torch.long).cumsum(0)
        self.register_buffer("offsets", offsets)

        # 2. 创建统一的大 Embedding 表
        total_vocab_size = sum(sizes)
        self.embedding = Embedding(total_vocab_size, embed_dim)

        # 3. 投影层
        total_embed_dim = len(sizes) * embed_dim
        self.projection = Linear(total_embed_dim, output_dim)

    def forward(self, categorical_features: Tensor) -> Tensor:
        """计算类别特征嵌入。

        Args:
            categorical_features: (batch, seq_len, num_features) 类别特征张量
                                  特征顺序需与 feature_names 一致

        Returns:
            (batch, seq_len, output_dim) 嵌入向量
        """
        # 确保输入为长整型
        if categorical_features.dtype != torch.long:
            categorical_features = categorical_features.long()

        # 应用 offsets 将多特征索引映射到单一 Embedding 空间
        # 利用广播机制: (B, L, F) + (F,) -> (B, L, F)
        x = categorical_features + self.offsets

        # 查表
        # (B, L, F) -> (B, L, F, D)
        emb = self.embedding(x)

        # 展平特征维度并投影
        # (B, L, F, D) -> (B, L, F*D)
        emb = rearrange(emb, "b l f d -> b l (f d)")

        return self.projection(emb)


class NumericalProjection(Module):
    """数值特征投影层。

    将数值特征向量通过 MLP 投影到目标维度。
    """

    def __init__(
        self,
        num_features: int,
        output_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        """初始化数值投影层。

        Args:
            num_features: 输入数值特征数量
            output_dim: 输出维度
            hidden_dim: 隐藏层维度，默认为 output_dim
            dropout: Dropout 概率
        """
        super().__init__()
        hidden_dim = hidden_dim or output_dim

        self.projection = Sequential(
            Linear(num_features, hidden_dim),
            GELU(),
            Dropout(dropout),
            Linear(hidden_dim, output_dim),
        )

    def forward(self, numerical_features: Tensor) -> Tensor:
        """投影数值特征。

        Args:
            numerical_features: (batch, seq_len, num_features) 数值特征

        Returns:
            (batch, seq_len, output_dim) 投影后的特征
        """
        return self.projection(numerical_features)

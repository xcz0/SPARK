"""输入层：特征融合模块。

- 数值特征流：通过 MLP 投影至 d_model/2
- 类别特征流：Embedding 后投影至 d_model/2
- 时间编码：Time2Vec 编码
- 最终融合：E_t = LayerNorm(Concat[h_num, h_cat]) + h_time
"""

import torch
from torch.nn import Module, Dropout, LayerNorm
from torch import Tensor

from .embeddings import CategoricalEmbedding, Time2Vec, NumericalProjection


class InputLayer(Module):
    """输入层：融合数值、类别和时间特征。

    融合公式：E_t = LayerNorm(Concat[h_num, h_cat]) + h_time
    """

    def __init__(
        self,
        d_model: int,
        num_numerical_features: int,
        categorical_vocab_sizes: dict[str, int],
        categorical_embed_dim: int = 16,
        dropout: float = 0.1,
    ):
        """初始化输入层。

        Args:
            d_model: 模型维度
            num_numerical_features: 数值特征数量
            categorical_vocab_sizes: 类别特征词表大小字典
            categorical_embed_dim: 每个类别特征的嵌入维度
            dropout: Dropout 概率
        """
        super().__init__()
        self.d_model = d_model

        # 确保数值流和类别流拼接后维度等于 d_model (处理 d_model 为奇数的情况)
        num_output_dim = d_model // 2
        cat_output_dim = d_model - num_output_dim

        # 数值特征投影：投影到 num_output_dim
        self.numerical_proj = NumericalProjection(
            num_features=num_numerical_features,
            output_dim=num_output_dim,
            dropout=dropout,
        )

        # 类别特征嵌入：投影到 cat_output_dim
        self.categorical_emb = CategoricalEmbedding(
            vocab_sizes=categorical_vocab_sizes,
            embed_dim=categorical_embed_dim,
            output_dim=cat_output_dim,
        )

        # Time2Vec 时间编码
        self.time2vec = Time2Vec(embed_dim=d_model)

        # LayerNorm
        self.layer_norm = LayerNorm(d_model)
        self.dropout = Dropout(dropout)

    def forward(
        self,
        numerical_features: Tensor,
        categorical_features: Tensor,
        time_stamps: Tensor,
    ) -> Tensor:
        """融合特征生成输入嵌入。

        Args:
            numerical_features: (batch, seq_len, num_numerical) 数值特征
            categorical_features: (batch, seq_len, num_categorical) 类别特征
            time_stamps: (batch, seq_len) 时间戳

        Returns:
            (batch, seq_len, d_model) 融合后的输入嵌入
        """
        # 数值特征投影
        h_num = self.numerical_proj(numerical_features)  # (batch, seq_len, d_model/2)

        # 类别特征嵌入
        h_cat = self.categorical_emb(
            categorical_features
        )  # (batch, seq_len, d_model/2)

        # 拼接数值和类别特征
        h_concat = torch.cat([h_num, h_cat], dim=-1)  # (batch, seq_len, d_model)

        # LayerNorm
        h_norm = self.layer_norm(h_concat)

        # Time2Vec 编码
        h_time = self.time2vec(time_stamps)  # (batch, seq_len, d_model)

        # 加法融合（类似位置编码）
        output = h_norm + h_time
        return self.dropout(output)

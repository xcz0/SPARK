"""Time-Aware Multi-Head Causal Transformer 架构。

采用 Decoder-only Transformer 结构：
- 输入层：数值/类别特征融合 + Time2Vec 时间编码
- Transformer 层：多头差异化注意力机制 (Pre-LayerNorm)
- 输出层：CORAL 评分预测 + 耗时预测
"""

from dataclasses import dataclass

from torch.nn import Module, ModuleList, LayerNorm, Linear, Dropout, GELU, Sequential
from torch import Tensor

from .components.attention import DifferentialMultiHeadAttention
from .components.heads import CORALHead, DurationHead
from .components.input_layer import InputLayer


@dataclass
class ModelConfig:
    """模型配置。"""

    d_model: int = 128
    n_heads: int = 8
    depth: int = 4
    d_ff: int | None = None
    dropout: float = 0.1
    num_numerical_features: int = 6
    categorical_vocab_sizes: dict[str, int] | None = None
    categorical_embed_dim: int = 16
    num_rating_classes: int = 4
    card_head_ratio: float = 0.5
    deck_head_ratio: float = 0.25
    numerical_feature_names: tuple[str, ...] | None = None
    time_feature_name: str = "day_offset"

    def __post_init__(self):
        if self.categorical_vocab_sizes is None:
            self.categorical_vocab_sizes = {
                "state": 4,
                "prev_review_rating": 5,
                "is_first_review": 2,
            }
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model


class TransformerBlock(Module):
    """Transformer 解码器块 (Pre-LayerNorm 结构)。

    包含：
    - 多头差异化注意力
    - 前馌网络
    - 残差连接和 Pre-LayerNorm

    使用 Pre-LN 结构：x + Attention(LN(x))，比 Post-LN 更稳定。
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: int | None = None,
        dropout: float = 0.1,
        card_head_ratio: float = 0.5,
        deck_head_ratio: float = 0.25,
    ):
        """初始化 Transformer 块。

        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff: 前馈网络隐藏维度，默认为 4 * d_model
            dropout: Dropout 概率
            card_head_ratio: 卡片头占比
            deck_head_ratio: 卡组头占比
        """
        super().__init__()
        d_ff = d_ff or 4 * d_model

        # 多头差异化注意力
        self.attention = DifferentialMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            card_head_ratio=card_head_ratio,
            deck_head_ratio=deck_head_ratio,
        )

        # 前馈网络
        self.feed_forward = Sequential(
            Linear(d_model, d_ff),
            GELU(),
            Dropout(dropout),
            Linear(d_ff, d_model),
            Dropout(dropout),
        )

        # LayerNorm
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        causal_mask: Tensor,
        card_mask: Tensor,
        deck_mask: Tensor,
        time_diff: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """前向传播。

        Args:
            x: (batch, seq_len, d_model) 输入
            causal_mask: (seq_len, seq_len) 因果掩码
            card_mask: (batch, seq_len, seq_len) 卡片掩码
            deck_mask: (batch, seq_len, seq_len) 卡组掩码
            time_diff: (batch, seq_len, seq_len) 时间差矩阵
            padding_mask: (batch, seq_len) 填充掩码

        Returns:
            (batch, seq_len, d_model) 输出
        """
        # Pre-LN: 先 LayerNorm，再注意力，然后残差
        attn_out = self.attention(
            self.norm1(x), causal_mask, card_mask, deck_mask, time_diff, padding_mask
        )
        x = x + self.dropout(attn_out)

        # Pre-LN: 先 LayerNorm，再 FFN，然后残差
        x = x + self.feed_forward(self.norm2(x))
        return x


class SPARKModel(Module):
    """SPARK: Spaced Repetition with Attention-based Recall Knowledge model.

    Time-Aware Multi-Head Causal Transformer 用于间隔重复学习预测。
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        d_model: int | None = None,
        n_heads: int | None = None,
        depth: int | None = None,
        d_ff: int | None = None,
        dropout: float | None = None,
        num_numerical_features: int | None = None,
        categorical_vocab_sizes: dict[str, int] | None = None,
        categorical_embed_dim: int | None = None,
        num_rating_classes: int | None = None,
        card_head_ratio: float | None = None,
        deck_head_ratio: float | None = None,
        numerical_feature_names: tuple[str, ...] | None = None,
        time_feature_name: str | None = None,
    ):
        """初始化模型并支持显式字段覆盖。"""

        super().__init__()

        overrides = {
            "d_model": d_model,
            "n_heads": n_heads,
            "depth": depth,
            "d_ff": d_ff,
            "dropout": dropout,
            "num_numerical_features": num_numerical_features,
            "categorical_vocab_sizes": categorical_vocab_sizes,
            "categorical_embed_dim": categorical_embed_dim,
            "num_rating_classes": num_rating_classes,
            "card_head_ratio": card_head_ratio,
            "deck_head_ratio": deck_head_ratio,
            "numerical_feature_names": numerical_feature_names,
            "time_feature_name": time_feature_name,
        }

        config_obj = config or ModelConfig()
        for field_name, value in overrides.items():
            if value is not None:
                setattr(config_obj, field_name, value)
        self.config = config_obj

        # 输入层
        self.input_layer = InputLayer(
            d_model=config_obj.d_model,
            num_numerical_features=config_obj.num_numerical_features,
            categorical_vocab_sizes=config_obj.categorical_vocab_sizes,
            categorical_embed_dim=config_obj.categorical_embed_dim,
            dropout=config_obj.dropout,
        )

        # Transformer 层堆叠
        self.transformer_blocks = ModuleList(
            [
                TransformerBlock(
                    d_model=config_obj.d_model,
                    n_heads=config_obj.n_heads,
                    d_ff=config_obj.d_ff,
                    dropout=config_obj.dropout,
                    card_head_ratio=config_obj.card_head_ratio,
                    deck_head_ratio=config_obj.deck_head_ratio,
                )
                for _ in range(config_obj.depth)
            ]
        )

        # Pre-LN 结构需要在最后添加 LayerNorm
        self.final_norm = LayerNorm(config_obj.d_model)

        # 输出头
        self.rating_head = CORALHead(
            d_model=config_obj.d_model,
            num_classes=config_obj.num_rating_classes,
        )
        self.duration_head = DurationHead(
            d_model=config_obj.d_model,
            dropout=config_obj.dropout,
        )

    def forward(
        self,
        numerical_features: Tensor,
        categorical_features: Tensor,
        time_stamps: Tensor,
        causal_mask: Tensor,
        card_mask: Tensor,
        deck_mask: Tensor,
        time_diff: Tensor,
        padding_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """前向传播。

        Args:
            numerical_features: (batch, seq_len, num_numerical) 数值特征
            categorical_features: (batch, seq_len, num_categorical) 类别特征
            time_stamps: (batch, seq_len) 时间戳
            causal_mask: (seq_len, seq_len) 因果掩码
            card_mask: (batch, seq_len, seq_len) 卡片掩码
            deck_mask: (batch, seq_len, seq_len) 卡组掩码
            time_diff: (batch, seq_len, seq_len) 时间差矩阵
            padding_mask: (batch, seq_len) 填充掩码

        Returns:
            包含以下键的字典:
            - rating_probs: (batch, seq_len, num_thresholds) 评分累积概率
            - duration_pred: (batch, seq_len) 耗时预测
            - hidden_states: (batch, seq_len, d_model) 最终隐藏状态
        """
        # 输入嵌入
        x = self.input_layer(numerical_features, categorical_features, time_stamps)

        # Transformer 层
        for block in self.transformer_blocks:
            x = block(x, causal_mask, card_mask, deck_mask, time_diff, padding_mask)

        # Pre-LN 结构的最终 LayerNorm
        x = self.final_norm(x)

        # 输出预测
        rating_probs = self.rating_head(x)
        duration_pred = self.duration_head(x)

        return {
            "rating_probs": rating_probs,
            "duration_pred": duration_pred,
            "hidden_states": x,
        }

    def predict_rating(self, rating_probs: Tensor) -> Tensor:
        """根据已计算的 rating 概率预测离散评分。"""

        return 1 + (rating_probs > 0.5).sum(dim=-1)

    def predict_expected_rating(self, rating_probs: Tensor) -> Tensor:
        """根据已计算的 rating 概率计算期望评分。"""

        return 1.0 + rating_probs.sum(dim=-1)

    def predict_correct(self, rating_probs: Tensor) -> Tensor:
        """根据已计算的 rating 概率预测回忆正确（评分大于1）的概率。"""

        return rating_probs[..., 0]

    @property
    def numerical_feature_names(self) -> tuple[str, ...] | None:
        """返回数值特征名称列表。

        Returns:
            如果配置中指定了特征名称则返回，否则返回 None。
        """
        return self.config.numerical_feature_names

    @property
    def categorical_feature_names(self) -> tuple[str, ...]:
        """返回类别特征名称列表（基于配置的词表键）。"""
        return tuple(self.config.categorical_vocab_sizes.keys())

    @property
    def time_feature_name(self) -> str:
        """返回时间特征名称。"""
        return self.config.time_feature_name

    @property
    def all_feature_names(self) -> dict[str, tuple[str, ...] | str | None]:
        """返回所有输入特征的名称。

        Returns:
            包含以下键的字典:
            - numerical: 数值特征名称元组（可能为 None）
            - categorical: 类别特征名称元组
            - time: 时间特征名称
        """
        return {
            "numerical": self.numerical_feature_names,
            "categorical": self.categorical_feature_names,
            "time": self.time_feature_name,
        }

"""Model subcomponents package.

包含模型的各个组件：
- embeddings: Time2Vec 时间编码和类别嵌入及数值映射
- input_layer: 输入特征融合层
- attention: 多头差异化注意力机制
- heads: 输出预测头
"""

from .attention import DifferentialMultiHeadAttention, TimeDecayBias
from .embeddings import CategoricalEmbedding, Time2Vec, NumericalProjection
from .heads import CORALHead, DurationHead
from .input_layer import InputLayer

__all__ = [
    # Embeddings
    "Time2Vec",
    "CategoricalEmbedding",
    # Input Layer
    "InputLayer",
    "NumericalProjection",
    # Attention
    "DifferentialMultiHeadAttention",
    "TimeDecayBias",
    # Output Heads
    "CORALHead",
    "DurationHead",
]

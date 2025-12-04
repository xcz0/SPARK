from typing import Any, ClassVar

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from src.data.config import CollatorConfig
from src.utils.masking import (
    create_card_mask,
    create_causal_mask,
    create_deck_mask,
    create_padding_mask,
    create_padding_mask_2d,
    create_time_diff_matrix,
)

type BatchType = list[dict[str, Tensor]]
type CollatorOutputType = dict[str, Tensor]


class ReviewCollator:
    """复习序列的 Collator。"""

    # 序列填充配置：(填充值, 数据类型)
    SEQUENCE_PAD_CONFIG: ClassVar[dict[str, tuple[Any, torch.dtype]]] = {
        "numerical_features": (0.0, torch.float32),
        "categorical_features": (0, torch.long),
        "time_stamps": (0.0, torch.float32),
        "card_ids": (-1, torch.long),
        "deck_ids": (-1, torch.long),
        "ordinal_targets": (0.0, torch.float32),
        "duration_targets": (0.0, torch.float32),
    }

    def __init__(self, config: CollatorConfig | None = None):
        self.config = config or CollatorConfig()

    def _pad_sequences(self, batch: BatchType) -> tuple[dict[str, Tensor], Tensor, int]:
        """填充序列并返回处理后的数据字典。"""
        seq_lens = torch.tensor([x["seq_len"] for x in batch], dtype=torch.long)
        max_len = int(seq_lens.max().item())

        first_sample = batch[0]
        padded: dict[str, Tensor] = {}

        for key, (pad_val, dtype) in self.SEQUENCE_PAD_CONFIG.items():
            if key not in first_sample:
                continue

            sequences = [s[key] for s in batch]

            # 仅在类型不匹配时转换
            if sequences[0].dtype != dtype:
                sequences = [s.to(dtype) for s in sequences]

            padded[key] = pad_sequence(
                sequences,
                batch_first=True,
                padding_value=pad_val,
            )

        return padded, seq_lens, max_len

    def _create_masks(
        self, padded: dict[str, Tensor], seq_lens: Tensor, max_len: int
    ) -> dict[str, Tensor]:
        """生成所有注意力掩码。"""
        padding_mask = create_padding_mask(seq_lens, max_len)
        padding_mask_2d = create_padding_mask_2d(padding_mask)
        device = seq_lens.device

        # 复合掩码需要与 padding mask 结合，过滤掉填充位置
        return {
            "padding_mask": padding_mask,
            "causal_mask": create_causal_mask(max_len, device=device),
            "time_diff": create_time_diff_matrix(padded["time_stamps"]),
            "card_mask": create_card_mask(padded["card_ids"]) & padding_mask_2d,
            "deck_mask": create_deck_mask(padded["deck_ids"], padded["card_ids"])
            & padding_mask_2d,
        }

    def __call__(self, batch: BatchType) -> CollatorOutputType:
        """处理批次数据。"""
        if not batch:
            return {}

        padded, seq_lens, max_len = self._pad_sequences(batch)
        masks = self._create_masks(padded, seq_lens, max_len)

        # 合并字典
        return padded | {"seq_lens": seq_lens} | masks

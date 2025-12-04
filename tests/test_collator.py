"""测试 src.data.collator 模块。"""

import pytest
import torch

from src.data.collator import ReviewCollator
from src.data.config import CollatorConfig


class TestCollatorConfig:
    """测试 CollatorConfig。"""

    def test_default_values(self):
        """测试默认值。"""
        config = CollatorConfig()

        assert config.pad_value == 0.0
        assert config.pad_id == -1

    def test_frozen(self):
        """测试不可变性。"""
        config = CollatorConfig()

        with pytest.raises(Exception):  # FrozenInstanceError
            config.pad_value = 1.0


class TestReviewCollator:
    """测试 ReviewCollator。"""

    @pytest.fixture
    def sample_batch(self):
        """创建测试用的批次数据。"""
        return [
            {
                "numerical_features": torch.randn(3, 6),
                "categorical_features": torch.randint(0, 5, (3, 3)),
                "time_stamps": torch.tensor([0.0, 1.0, 2.0]),
                "card_ids": torch.tensor([1, 1, 2]),
                "deck_ids": torch.tensor([1, 1, 1]),
                "ordinal_targets": torch.rand(3, 3),
                "duration_targets": torch.rand(3),
                "seq_len": torch.tensor(3),
            },
            {
                "numerical_features": torch.randn(5, 6),
                "categorical_features": torch.randint(0, 5, (5, 3)),
                "time_stamps": torch.tensor([0.0, 0.5, 1.0, 2.0, 3.0]),
                "card_ids": torch.tensor([1, 2, 2, 1, 3]),
                "deck_ids": torch.tensor([1, 1, 1, 1, 2]),
                "ordinal_targets": torch.rand(5, 3),
                "duration_targets": torch.rand(5),
                "seq_len": torch.tensor(5),
            },
        ]

    @pytest.fixture
    def collator(self):
        """返回默认配置的 ReviewCollator。"""
        return ReviewCollator()

    @pytest.fixture
    def collated_batch(self, collator, sample_batch):
        """缓存经过默认 collator 的批次结果以避免重复计算。"""
        return collator(sample_batch)

    def test_output_keys(self, collated_batch):
        """测试输出包含所有必需的键。"""
        expected_keys = {
            "numerical_features",
            "categorical_features",
            "time_stamps",
            "card_ids",
            "deck_ids",
            "ordinal_targets",
            "duration_targets",
            "seq_lens",
            "padding_mask",
            "causal_mask",
            "card_mask",
            "deck_mask",
            "time_diff",
        }
        assert set(collated_batch.keys()) == expected_keys

    def test_padding_to_max_length(self, collated_batch):
        """测试填充到最大长度。"""
        # 最大长度应为 5
        assert collated_batch["numerical_features"].shape == (2, 5, 6)
        assert collated_batch["padding_mask"].shape == (2, 5)
        assert collated_batch["card_mask"].shape == (2, 5, 5)

    def test_padding_mask_correctness(self, collated_batch):
        """测试填充掩码正确性。"""
        # 第一个样本长度 3，位置 3, 4 应为 False
        assert collated_batch["padding_mask"][0, :3].all()
        assert not collated_batch["padding_mask"][0, 3:].any()

        # 第二个样本长度 5，全为 True
        assert collated_batch["padding_mask"][1].all()

    def test_causal_mask_shape(self, collated_batch):
        """测试因果掩码形状。"""
        # 因果掩码是 2D 的
        assert collated_batch["causal_mask"].dim() == 2
        assert collated_batch["causal_mask"].shape == (5, 5)

    def test_masks_respect_padding(self, collated_batch):
        """测试掩码尊重填充。"""
        # 第一个样本填充位置的掩码应为 False
        assert not collated_batch["card_mask"][0, 0, 3].item()
        assert not collated_batch["card_mask"][0, 3, 0].item()
        assert not collated_batch["deck_mask"][0, 0, 4].item()

    def test_seq_lens_preserved(self, collated_batch):
        """测试序列长度保持。"""
        assert torch.equal(collated_batch["seq_lens"], torch.tensor([3, 5]))

    def test_custom_config(self, sample_batch):
        """测试自定义配置。"""
        config = CollatorConfig(pad_value=1.0, pad_id=0)
        collator = ReviewCollator(config)
        result = collator(sample_batch)

        # 确保可以正常运行
        assert result["numerical_features"].shape[0] == 2

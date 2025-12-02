"""测试 src.utils.masking 模块中的掩码工具函数。"""

import pytest
import torch

from src.utils.masking import (
    apply_padding_to_attention_mask,
    combine_masks,
    create_card_mask,
    create_causal_mask,
    create_deck_mask,
    create_different_element_mask,
    create_padding_mask,
    create_padding_mask_2d,
    create_same_element_mask,
    create_time_diff_matrix,
)

# ============================================================================
# 共享 Fixtures
# ============================================================================

CPU_DEVICE = torch.device("cpu")


# ============================================================================
# 基础掩码函数测试
# ============================================================================


class TestCreatePaddingMask:
    """测试 create_padding_mask 函数。"""

    @pytest.mark.parametrize(
        "seq_lens, max_len, expected",
        [
            # 基本多序列
            ([3, 5, 2], 5, [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0]]),
            # 单序列
            ([3], 5, [[1, 1, 1, 0, 0]]),
            # 全长序列
            ([4, 4, 4], 4, [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]),
        ],
    )
    def test_padding_mask_values(self, seq_lens, max_len, expected):
        """测试填充掩码的值。"""
        mask = create_padding_mask(torch.tensor(seq_lens), max_len=max_len)
        assert torch.equal(mask, torch.tensor(expected, dtype=torch.bool))

    def test_auto_max_len(self):
        """测试自动推断最大长度。"""
        seq_lens = torch.tensor([2, 4, 3])
        mask = create_padding_mask(seq_lens, max_len=None)

        assert mask.shape == (3, 4)
        assert mask.dtype == torch.bool
        assert mask[1].all()  # 长度为 4 的序列全为 True


class TestCreateCausalMask:
    """测试 create_causal_mask 函数。"""

    def test_causal_mask_structure(self):
        """测试因果掩码的形状和下三角性质。"""
        mask = create_causal_mask(4, CPU_DEVICE)

        assert mask.shape == (4, 4)
        assert mask.dtype == torch.bool
        # 验证是下三角矩阵
        expected = torch.tril(torch.ones(4, 4, dtype=torch.bool))
        assert torch.equal(mask, expected)


class TestElementMasks:
    """测试 same_element 和 different_element 掩码函数。"""

    def test_same_element_basic(self):
        """测试同元素掩码的基本行为。"""
        ids = torch.tensor([[1, 1, 2, 1]])
        mask = create_same_element_mask(ids)

        expected = torch.tensor(
            [
                [[True, True, False, True]] * 2
                + [[False, False, True, False]]
                + [[True, True, False, True]]
            ]
        )
        assert torch.equal(mask, expected)

    def test_same_element_all_same(self):
        """测试全同元素情况。"""
        mask = create_same_element_mask(torch.tensor([[5, 5, 5]]))
        assert mask.all()

    def test_same_and_different_are_complement(self):
        """验证 same 和 different 掩码互补。"""
        ids = torch.tensor([[1, 2, 1, 3]])
        same_mask = create_same_element_mask(ids)
        diff_mask = create_different_element_mask(ids)
        assert torch.equal(same_mask, ~diff_mask)


class TestCombineMasks:
    """测试 combine_masks 函数。"""

    def test_combine_two_masks(self):
        """测试两个掩码的组合（逻辑与）。"""
        mask1 = torch.tensor([[True, True], [True, False]], dtype=torch.bool)
        mask2 = torch.tensor([[True, False], [True, True]], dtype=torch.bool)
        result = combine_masks(mask1, mask2)
        expected = torch.tensor([[True, False], [True, False]], dtype=torch.bool)
        assert torch.equal(result, expected)

    def test_broadcasting(self):
        """测试维度广播。"""
        mask_2d = torch.tensor([[True, False], [True, True]])
        mask_3d = torch.ones(2, 2, 2, dtype=torch.bool)

        result = combine_masks(mask_3d, mask_2d)
        assert result.shape == (2, 2, 2)


class TestPaddingMask2dAndApply:
    """测试 create_padding_mask_2d 和 apply_padding_to_attention_mask 函数。"""

    def test_2d_mask_basic(self):
        """测试 2D 掩码生成和对称性。"""
        padding_mask = torch.tensor([[True, True, False]])
        result = create_padding_mask_2d(padding_mask)

        assert result.shape == (1, 3, 3)
        # 验证对称性
        assert torch.equal(result, result.transpose(1, 2))
        # 验证有效/无效位置
        assert result[0, 0, 0].item() is True
        assert result[0, 0, 2].item() is False

    def test_apply_with_2d_attention_mask(self):
        """测试 2D 注意力掩码自动扩展。"""
        attention_mask = create_causal_mask(3, CPU_DEVICE)
        padding_mask = torch.tensor([[True, True, False], [True, True, True]])

        result = apply_padding_to_attention_mask(attention_mask, padding_mask)

        assert result.shape == (2, 3, 3)
        assert result[0, 2, 0].item() is False  # 因果 + 填充
        assert result[1, 2, 0].item() is True  # 纯因果

    def test_preserves_pattern_without_padding(self):
        """测试无填充时保留原始注意力模式。"""
        attention_mask = create_causal_mask(4, CPU_DEVICE)
        padding_mask = torch.ones(1, 4, dtype=torch.bool)

        result = apply_padding_to_attention_mask(attention_mask, padding_mask)
        assert torch.equal(result[0], attention_mask)


# ============================================================================
# 复合掩码函数测试
# ============================================================================


class TestCreateCardMask:
    """测试 create_card_mask 函数。"""

    def test_basic_card_mask(self):
        """测试卡片掩码：同卡片 ∩ 因果。"""
        # 卡片序列：[1, 1, 2, 1] - 位置0,1,3是卡片1，位置2是卡片2
        card_ids = torch.tensor([[1, 1, 2, 1]])
        mask = create_card_mask(card_ids)

        # 验证关键位置
        assert mask[0, 1, 0].item() is True  # 位置1看位置0（同卡片）
        assert mask[0, 2, 0].item() is False  # 位置2看位置0（不同卡片）
        assert mask[0, 3, 0].item() is True  # 位置3看位置0（同卡片）
        assert mask[0, 3, 2].item() is False  # 位置3看位置2（不同卡片）
        assert mask[0, 0, 1].item() is False  # 位置0看位置1（因果阻止）

    def test_all_same_card_equals_causal(self):
        """全同卡片时退化为因果掩码。"""
        card_ids = torch.tensor([[5, 5, 5, 5]])
        mask = create_card_mask(card_ids)
        causal = torch.tril(torch.ones(4, 4, dtype=torch.bool))
        assert torch.equal(mask[0], causal)


class TestCreateDeckMask:
    """测试 create_deck_mask 函数。"""

    def test_basic_deck_mask(self):
        """测试卡组掩码：同卡组 ∩ 不同卡片 ∩ 因果。"""
        # 卡组[1,1,1]，卡片[A,B,A]
        deck_ids = torch.tensor([[1, 1, 1]])
        card_ids = torch.tensor([[1, 2, 1]])
        mask = create_deck_mask(deck_ids, card_ids)

        assert mask[0, 0, 0].item() is False  # 同卡片被排除
        assert mask[0, 1, 0].item() is True  # 同卡组不同卡片
        assert mask[0, 2, 1].item() is True  # 同卡组不同卡片
        assert mask[0, 2, 0].item() is False  # 同卡片被排除

    def test_different_decks_isolation(self):
        """不同卡组隔离。"""
        deck_ids = torch.tensor([[1, 2, 1]])
        card_ids = torch.tensor([[1, 2, 3]])
        mask = create_deck_mask(deck_ids, card_ids)

        assert mask[0, 2, 0].item() is True  # 同卡组不同卡片
        assert mask[0, 2, 1].item() is False  # 不同卡组


class TestCreateTimeDiffMatrix:
    """测试 create_time_diff_matrix 函数。"""

    def test_basic_time_diff(self):
        """测试时间差计算正确性。"""
        time_stamps = torch.tensor([[0.0, 1.0, 3.0]])
        result = create_time_diff_matrix(time_stamps)

        expected = torch.tensor([[[0.0, 1.0, 3.0], [1.0, 0.0, 2.0], [3.0, 2.0, 0.0]]])
        assert torch.allclose(result, expected)

    def test_matrix_properties(self):
        """测试矩阵性质：对称性和对角线为零。"""
        time_stamps = torch.tensor([[1.0, 2.5, 4.0, 7.0]])
        result = create_time_diff_matrix(time_stamps)

        # 对称性
        assert torch.allclose(result, result.transpose(1, 2))
        # 对角线为零
        diagonal = torch.diagonal(result, dim1=1, dim2=2)
        assert torch.allclose(diagonal, torch.zeros_like(diagonal))

"""测试 src.models.losses 模块。"""

import pytest
import torch

from src.models.losses import (
    CORALLoss,
    CombinedLoss,
    DurationLoss,
    _reduce_loss,
)
from src.models import CoralOrdinalLoss  # 从 __init__ 导入向后兼容别名


class TestReduceLoss:
    """测试 _reduce_loss 辅助函数。"""

    def test_mean_reduction_no_mask(self):
        """测试无掩码的 mean 聚合。"""
        loss = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = _reduce_loss(loss, None, "mean")

        assert result.item() == pytest.approx(2.5)

    def test_sum_reduction_no_mask(self):
        """测试无掩码的 sum 聚合。"""
        loss = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = _reduce_loss(loss, None, "sum")

        assert result.item() == pytest.approx(10.0)

    def test_none_reduction_no_mask(self):
        """测试无掩码的 none 聚合。"""
        loss = torch.tensor([1.0, 2.0, 3.0])
        result = _reduce_loss(loss, None, "none")

        assert torch.equal(result, loss)

    def test_mean_reduction_with_mask(self):
        """测试有掩码的 mean 聚合。"""
        loss = torch.tensor([1.0, 2.0, 100.0, 100.0])
        mask = torch.tensor([True, True, False, False])
        result = _reduce_loss(loss, mask, "mean")

        # (1.0 + 2.0) / 2 = 1.5
        assert result.item() == pytest.approx(1.5)

    def test_sum_reduction_with_mask(self):
        """测试有掩码的 sum 聚合。"""
        loss = torch.tensor([1.0, 2.0, 100.0])
        mask = torch.tensor([True, True, False])
        result = _reduce_loss(loss, mask, "sum")

        assert result.item() == pytest.approx(3.0)

    def test_none_reduction_with_mask(self):
        """测试有掩码的 none 聚合。"""
        loss = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([True, False, True])
        result = _reduce_loss(loss, mask, "none")

        expected = torch.tensor([1.0, 0.0, 3.0])
        assert torch.allclose(result, expected)

    def test_empty_mask_mean(self):
        """测试全空掩码时的安全处理。"""
        loss = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([False, False, False])
        result = _reduce_loss(loss, mask, "mean")

        # 应返回 0 而不是 nan
        assert result.item() == pytest.approx(0.0)


class TestCORALLoss:
    """测试 CORALLoss 类。"""

    @pytest.fixture
    def loss_fn(self):
        """创建损失函数实例。"""
        return CORALLoss(reduction="mean")

    def test_perfect_predictions(self, loss_fn):
        """测试完美预测。"""
        # 目标全为 1 (y > k 全为真)
        probs = torch.ones(2, 4, 3)
        targets = torch.ones(2, 4, 3)

        loss = loss_fn(probs, targets)

        # BCE(1, 1) ≈ 0
        assert loss.item() < 0.1

    def test_worst_predictions(self, loss_fn):
        """测试最差预测。"""
        # 预测与目标完全相反
        probs = torch.zeros(2, 4, 3)
        targets = torch.ones(2, 4, 3)

        loss = loss_fn(probs, targets)

        # BCE(0, 1) 很大
        assert loss.item() > 10.0

    def test_output_shape(self, loss_fn):
        """测试输出形状。"""
        probs = torch.sigmoid(torch.randn(2, 4, 3))
        targets = torch.rand(2, 4, 3)

        loss = loss_fn(probs, targets)

        # mean 聚合应返回标量
        assert loss.dim() == 0

    def test_with_mask(self, loss_fn):
        """测试带掩码。"""
        probs = torch.sigmoid(torch.randn(2, 4, 3))
        targets = torch.rand(2, 4, 3)
        mask = torch.tensor([[True, True, False, False], [True, True, True, False]])

        loss = loss_fn(probs, targets, mask)

        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_reduction_none(self):
        """测试 none 聚合。"""
        loss_fn = CORALLoss(reduction="none")
        probs = torch.sigmoid(torch.randn(2, 4, 3))
        targets = torch.rand(2, 4, 3)

        loss = loss_fn(probs, targets)

        # 返回 (batch, seq_len) 形状
        assert loss.shape == (2, 4)

    def test_reduction_sum(self):
        """测试 sum 聚合。"""
        loss_fn = CORALLoss(reduction="sum")
        probs = torch.sigmoid(torch.randn(2, 4, 3))
        targets = torch.rand(2, 4, 3)

        loss = loss_fn(probs, targets)

        assert loss.dim() == 0

    def test_backward(self, loss_fn):
        """测试反向传播。"""
        probs = torch.sigmoid(torch.randn(2, 4, 3, requires_grad=True))
        targets = torch.rand(2, 4, 3)

        loss = loss_fn(probs, targets)
        loss.backward()

        assert probs.grad is not None
        assert probs.grad.shape == probs.shape


class TestDurationLoss:
    """测试 DurationLoss 类。"""

    @pytest.fixture
    def loss_fn(self):
        """创建损失函数实例。"""
        return DurationLoss(reduction="mean")

    def test_perfect_predictions(self, loss_fn):
        """测试完美预测。"""
        preds = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        targets = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        loss = loss_fn(preds, targets)

        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_basic_mse(self, loss_fn):
        """测试基本 MSE 计算。"""
        preds = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        targets = torch.tensor([[1.0, 1.0], [1.0, 1.0]])

        loss = loss_fn(preds, targets)

        assert loss.item() == pytest.approx(1.0)

    def test_with_mask(self, loss_fn):
        """测试带掩码。"""
        preds = torch.tensor([[0.0, 0.0, 100.0]])
        targets = torch.tensor([[1.0, 1.0, 0.0]])
        mask = torch.tensor([[True, True, False]])

        loss = loss_fn(preds, targets, mask)

        # 只考虑前两个位置
        assert loss.item() == pytest.approx(1.0)

    def test_reduction_none(self):
        """测试 none 聚合。"""
        loss_fn = DurationLoss(reduction="none")
        preds = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        targets = torch.tensor([[1.0, 1.0], [1.0, 1.0]])

        loss = loss_fn(preds, targets)

        expected = torch.tensor([[1.0, 0.0], [1.0, 4.0]])
        assert torch.allclose(loss, expected)

    def test_reduction_sum(self):
        """测试 sum 聚合。"""
        loss_fn = DurationLoss(reduction="sum")
        preds = torch.zeros(2, 3)
        targets = torch.ones(2, 3)

        loss = loss_fn(preds, targets)

        assert loss.item() == pytest.approx(6.0)

    def test_backward(self, loss_fn):
        """测试反向传播。"""
        preds = torch.randn(2, 4, requires_grad=True)
        targets = torch.randn(2, 4)

        loss = loss_fn(preds, targets)
        loss.backward()

        assert preds.grad is not None


class TestCombinedLoss:
    """测试 CombinedLoss 类。"""

    @pytest.fixture
    def loss_fn(self):
        """创建损失函数实例。"""
        return CombinedLoss(rating_weight=1.0, duration_weight=0.1)

    def test_output_keys(self, loss_fn):
        """测试输出包含所需的键。"""
        probs = torch.sigmoid(torch.randn(2, 4, 3))
        ordinal_targets = torch.rand(2, 4, 3)
        duration_pred = torch.randn(2, 4)
        duration_target = torch.randn(2, 4)

        result = loss_fn(probs, ordinal_targets, duration_pred, duration_target)

        assert "total" in result
        assert "rating" in result
        assert "duration" in result

    def test_total_is_weighted_sum(self, loss_fn):
        """测试总损失是加权和。"""
        probs = torch.sigmoid(torch.randn(2, 4, 3))
        ordinal_targets = torch.rand(2, 4, 3)
        duration_pred = torch.randn(2, 4)
        duration_target = torch.randn(2, 4)

        result = loss_fn(probs, ordinal_targets, duration_pred, duration_target)

        expected_total = 1.0 * result["rating"] + 0.1 * result["duration"]
        assert result["total"].item() == pytest.approx(expected_total.item(), rel=1e-5)

    def test_custom_weights(self):
        """测试自定义权重。"""
        loss_fn = CombinedLoss(rating_weight=2.0, duration_weight=0.5)

        probs = torch.sigmoid(torch.randn(2, 4, 3))
        ordinal_targets = torch.rand(2, 4, 3)
        duration_pred = torch.randn(2, 4)
        duration_target = torch.randn(2, 4)

        result = loss_fn(probs, ordinal_targets, duration_pred, duration_target)

        expected_total = 2.0 * result["rating"] + 0.5 * result["duration"]
        assert result["total"].item() == pytest.approx(expected_total.item(), rel=1e-5)

    def test_with_mask(self, loss_fn):
        """测试带掩码。"""
        probs = torch.sigmoid(torch.randn(2, 4, 3))
        ordinal_targets = torch.rand(2, 4, 3)
        duration_pred = torch.randn(2, 4)
        duration_target = torch.randn(2, 4)
        mask = torch.tensor([[True, True, False, False], [True, True, True, False]])

        result = loss_fn(probs, ordinal_targets, duration_pred, duration_target, mask)

        assert not torch.isnan(result["total"])
        assert not torch.isnan(result["rating"])
        assert not torch.isnan(result["duration"])

    def test_backward(self, loss_fn):
        """测试反向传播。"""
        probs = torch.sigmoid(torch.randn(2, 4, 3, requires_grad=True))
        ordinal_targets = torch.rand(2, 4, 3)
        duration_pred = torch.randn(2, 4, requires_grad=True)
        duration_target = torch.randn(2, 4)

        result = loss_fn(probs, ordinal_targets, duration_pred, duration_target)
        result["total"].backward()

        assert probs.grad is not None
        assert duration_pred.grad is not None

    def test_weights_as_buffers(self):
        """测试权重作为 buffer 注册。"""
        loss_fn = CombinedLoss(rating_weight=1.5, duration_weight=0.2)

        # 检查 buffer 存在
        assert hasattr(loss_fn, "rating_weight")
        assert hasattr(loss_fn, "duration_weight")

        # 检查值正确
        assert loss_fn.rating_weight.item() == pytest.approx(1.5)
        assert loss_fn.duration_weight.item() == pytest.approx(0.2)

    def test_device_transfer(self):
        """测试设备转移。"""
        loss_fn = CombinedLoss()

        # buffer 应该跟随模块转移
        state_dict = loss_fn.state_dict()
        assert "rating_weight" in state_dict
        assert "duration_weight" in state_dict


class TestBackwardCompatibility:
    """测试向后兼容性。"""

    def test_coral_ordinal_loss_alias(self):
        """测试 CoralOrdinalLoss 别名。"""
        assert CoralOrdinalLoss is CORALLoss

        loss_fn = CoralOrdinalLoss()
        probs = torch.sigmoid(torch.randn(2, 4, 3))
        targets = torch.rand(2, 4, 3)

        loss = loss_fn(probs, targets)
        assert not torch.isnan(loss)


class TestLossDevicePlacement:
    """测试损失函数在不同设备上的行为。"""

    def test_coral_loss_cpu(self):
        """测试 CORALLoss 在 CPU 上。"""
        loss_fn = CORALLoss()
        probs = torch.sigmoid(torch.randn(2, 4, 3))
        targets = torch.rand(2, 4, 3)

        loss = loss_fn(probs, targets)
        assert loss.device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_coral_loss_cuda(self):
        """测试 CORALLoss 在 CUDA 上。"""
        loss_fn = CORALLoss().cuda()
        probs = torch.sigmoid(torch.randn(2, 4, 3)).cuda()
        targets = torch.rand(2, 4, 3).cuda()

        loss = loss_fn(probs, targets)
        assert loss.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_combined_loss_cuda(self):
        """测试 CombinedLoss 在 CUDA 上。"""
        loss_fn = CombinedLoss().cuda()
        probs = torch.sigmoid(torch.randn(2, 4, 3)).cuda()
        ordinal_targets = torch.rand(2, 4, 3).cuda()
        duration_pred = torch.randn(2, 4).cuda()
        duration_target = torch.randn(2, 4).cuda()

        result = loss_fn(probs, ordinal_targets, duration_pred, duration_target)
        assert result["total"].device.type == "cuda"

"""测试 src.utils.metrics 模块。"""

import pytest
import torch
from torchmetrics import MeanSquaredError

from src.utils.metrics import OrdinalAccuracy, ordinal_accuracy, rmse


class TestOrdinalAccuracy:
    """测试 OrdinalAccuracy 指标类。"""

    def test_perfect_predictions(self):
        """测试完美预测场景。"""
        metric = OrdinalAccuracy()
        preds = torch.tensor([1, 2, 3, 4])
        targets = torch.tensor([1, 2, 3, 4])

        metric.update(preds, targets)
        result = metric.compute()

        assert result.item() == pytest.approx(1.0)

    def test_all_wrong_predictions(self):
        """测试全部错误预测。"""
        metric = OrdinalAccuracy()
        preds = torch.tensor([4, 3, 2, 1])
        targets = torch.tensor([1, 2, 3, 4])

        metric.update(preds, targets)
        result = metric.compute()

        assert result.item() == pytest.approx(0.0)

    def test_partial_correct_predictions(self):
        """测试部分正确预测。"""
        metric = OrdinalAccuracy()
        preds = torch.tensor([1, 2, 4, 4])
        targets = torch.tensor([1, 2, 3, 4])

        metric.update(preds, targets)
        result = metric.compute()

        assert result.item() == pytest.approx(0.75)

    def test_with_cumulative_probs(self):
        """测试累积概率输入。"""
        metric = OrdinalAccuracy()
        # 累积概率 [P(y>1), P(y>2), P(y>3)]
        # 第一个: [0.9, 0.8, 0.2] -> 评分 3 (2 个阈值超过 0.5)
        # 第二个: [0.9, 0.1, 0.1] -> 评分 2 (1 个阈值超过 0.5)
        preds = torch.tensor([[0.9, 0.8, 0.2], [0.9, 0.1, 0.1]])
        targets = torch.tensor([3, 2])

        metric.update(preds, targets)
        result = metric.compute()

        assert result.item() == pytest.approx(1.0)

    def test_cumulative_probs_incorrect(self):
        """测试累积概率不正确预测。"""
        metric = OrdinalAccuracy()
        # [0.9, 0.8, 0.6] -> 评分 4
        # [0.1, 0.1, 0.1] -> 评分 1
        preds = torch.tensor([[0.9, 0.8, 0.6], [0.1, 0.1, 0.1]])
        targets = torch.tensor([1, 4])

        metric.update(preds, targets)
        result = metric.compute()

        assert result.item() == pytest.approx(0.0)

    def test_multiple_updates(self):
        """测试多次更新累积。"""
        metric = OrdinalAccuracy()

        # 第一批: 2/2 正确
        metric.update(torch.tensor([1, 2]), torch.tensor([1, 2]))
        # 第二批: 1/2 正确
        metric.update(torch.tensor([3, 4]), torch.tensor([3, 1]))

        result = metric.compute()
        assert result.item() == pytest.approx(0.75)

    def test_targets_with_threshold_encoding(self):
        """测试有序目标为阈值编码时的处理。"""
        metric = OrdinalAccuracy()

        preds = torch.tensor([3, 1])
        targets = torch.tensor([[1, 1, 0], [0, 0, 0]], dtype=torch.float32)

        metric.update(preds, targets)

        result = metric.compute()
        assert result.item() == pytest.approx(1.0)

    def test_reset(self):
        """测试重置功能。"""
        metric = OrdinalAccuracy()

        metric.update(torch.tensor([1, 2]), torch.tensor([1, 2]))
        metric.reset()
        metric.update(torch.tensor([1]), torch.tensor([2]))

        result = metric.compute()
        assert result.item() == pytest.approx(0.0)

    def test_empty_update(self):
        """测试空更新时的安全处理。"""
        metric = OrdinalAccuracy()
        result = metric.compute()

        # 应返回 0 而不是 nan
        assert result.item() == pytest.approx(0.0)


class TestRMSEFunctional:
    """测试 rmse 函数式接口。"""

    def test_basic(self):
        """测试基本 RMSE 计算。"""
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])

        result = rmse(preds, targets)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_with_error(self):
        """测试有误差的 RMSE。"""
        preds = torch.tensor([0.0, 0.0, 0.0, 0.0])
        targets = torch.tensor([1.0, 1.0, 1.0, 1.0])

        result = rmse(preds, targets)
        assert result.item() == pytest.approx(1.0, rel=1e-5)

    def test_float_precision(self):
        """测试浮点精度。"""
        preds = torch.tensor([1.5, 2.5, 3.5])
        targets = torch.tensor([1.0, 2.0, 3.0])

        result = rmse(preds, targets)
        assert result.item() == pytest.approx(0.5, rel=1e-5)


class TestOrdinalAccuracyFunctional:
    """测试 ordinal_accuracy 函数式接口。"""

    def test_basic(self):
        """测试基本准确率计算。"""
        preds = torch.tensor([1, 2, 3, 4])
        targets = torch.tensor([1, 2, 3, 4])

        result = ordinal_accuracy(preds, targets)
        assert result.item() == pytest.approx(1.0)

    def test_partial(self):
        """测试部分正确。"""
        preds = torch.tensor([1, 2, 3])
        targets = torch.tensor([1, 2, 4])

        result = ordinal_accuracy(preds, targets)
        assert result.item() == pytest.approx(2 / 3, rel=1e-5)

    def test_with_probs(self):
        """测试累积概率输入。"""
        # [0.9, 0.6, 0.1] -> 评分 3
        preds = torch.tensor([[0.9, 0.6, 0.1]])
        targets = torch.tensor([3])

        result = ordinal_accuracy(preds, targets)
        assert result.item() == pytest.approx(1.0)

    def test_all_wrong(self):
        """测试全部错误。"""
        preds = torch.tensor([1, 1, 1])
        targets = torch.tensor([2, 3, 4])

        result = ordinal_accuracy(preds, targets)
        assert result.item() == pytest.approx(0.0)

    def test_targets_with_threshold_encoding(self):
        """测试函数式接口可接受阈值编码目标。"""
        preds = torch.tensor([3, 2])
        targets = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)

        result = ordinal_accuracy(preds, targets)
        assert result.item() == pytest.approx(1.0)


class TestMetricDevicePlacement:
    """测试指标在不同设备上的行为。"""

    def test_ordinal_accuracy_cpu(self):
        """测试 OrdinalAccuracy 在 CPU 上。"""
        metric = OrdinalAccuracy()
        preds = torch.tensor([1, 2, 3])
        targets = torch.tensor([1, 2, 3])

        metric.update(preds, targets)
        result = metric.compute()

        assert result.device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_ordinal_accuracy_cuda(self):
        """测试 OrdinalAccuracy 在 CUDA 上。"""
        metric = OrdinalAccuracy().cuda()
        preds = torch.tensor([1, 2, 3]).cuda()
        targets = torch.tensor([1, 2, 3]).cuda()

        metric.update(preds, targets)
        result = metric.compute()

        assert result.device.type == "cuda"
